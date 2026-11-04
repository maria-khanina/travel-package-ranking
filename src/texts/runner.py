"""Runs one text generation job against an OpenAI-compatible server."""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
import pandas as pd

from src.settings.loader import Settings
from src.settings.records import PromptSettings
from src.texts.client import ExtraBody, build_client, generate_valid_text
from src.texts.details import collect_required_phrases
from src.texts.diversity import DiversityFilter, split_words
from src.texts.models import (
    JOB_HOTELS,
    JOB_INTENTS,
    JOB_PRACTICAL,
    JOB_ROOMS,
    HotelSpec,
    IntentSpec,
    PracticalSpec,
    RoomSpec,
    TextJobs,
    TextRecord,
)
from src.texts.prompts import (
    build_hotel_messages,
    build_intent_messages,
    build_practical_messages,
    build_room_messages,
)
from src.texts.store import (
    append_record,
    compact_records,
    open_for_append,
    read_text_map,
)
from src.texts.validation import TextValidator

THINKING_OFF: ExtraBody = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass
class JobRequest:
    job: str
    model: str
    limit: int | None
    output_path: Path
    disable_thinking: bool


class JobWriter:
    def __init__(self, path: Path, total: int, progress_every: int) -> None:
        self.handle = open_for_append(path)
        self.lock = asyncio.Lock()
        self.total = total
        self.progress_every = progress_every
        self.started = time.time()
        self.succeeded = 0
        self.failed = 0

    async def write(self, record: TextRecord) -> None:
        """Stores a record and prints progress periodically."""
        async with self.lock:
            append_record(self.handle, record)
            self.succeeded += record.text is not None
            self.failed += record.text is None
            done = self.succeeded + self.failed
            if done % self.progress_every == 0:
                print(self.describe(done), flush=True)

    def describe(self, done: int) -> str:
        """Describes the progress and the speed so far."""
        speed = done / (time.time() - self.started)
        counts = f"ok={self.succeeded} failed={self.failed}"
        return f"{done}/{self.total} {counts} {speed:.1f}/s"

    def close(self) -> None:
        """Closes the output file."""
        self.handle.close()


@dataclass
class JobContext:
    request: JobRequest
    client: AsyncOpenAI
    semaphore: asyncio.Semaphore
    writer: JobWriter
    validator: TextValidator
    diversity: DiversityFilter
    settings: Settings


def collect_place_words(settings: Settings) -> list[str]:
    """Lists the names of every destination city and country in lower case."""
    words: set[str] = set()
    for record in settings.destinations:
        words.add(record.city.lower())
        words.add(record.country.lower())
    return sorted(words)


def pick_validator(
    job: str,
    validator: TextValidator,
    spec_json: str,
) -> Callable[[str], bool]:
    """Chooses the validation rule that belongs to a brief of a job."""
    if job == JOB_INTENTS:
        limit = IntentSpec.model_validate_json(spec_json).maximum_words
        return partial(validator.validate_intent, maximum_words=limit)
    if job == JOB_PRACTICAL:
        limit = PracticalSpec.model_validate_json(spec_json).maximum_words
        return partial(validator.validate_practical, maximum_words=limit)
    if job == JOB_HOTELS:
        return validator.validate_hotel
    return validator.validate_room


def build_intent_job_messages(
    spec_json: str,
    prompts: PromptSettings,
) -> list[ChatCompletionMessageParam]:
    """Builds the chat messages of one traveler vibe brief."""
    spec = IntentSpec.model_validate_json(spec_json)
    return build_intent_messages(spec, prompts.intent, prompts)


def build_practical_job_messages(
    spec_json: str,
    prompts: PromptSettings,
) -> list[ChatCompletionMessageParam]:
    """Builds the chat messages of one practical requirements brief."""
    spec = PracticalSpec.model_validate_json(spec_json)
    return build_practical_messages(spec, prompts.practical, prompts)


def build_hotel_job_messages(
    spec_json: str,
    prompts: PromptSettings,
) -> list[ChatCompletionMessageParam]:
    """Builds the chat messages of one hotel description brief."""
    spec = HotelSpec.model_validate_json(spec_json)
    return build_hotel_messages(spec, prompts.hotel, prompts)


def build_room_job_messages(
    spec_json: str,
    prompts: PromptSettings,
) -> list[ChatCompletionMessageParam]:
    """Builds the chat messages of one room description brief."""
    spec = RoomSpec.model_validate_json(spec_json)
    return build_room_messages(spec, prompts.room, prompts)


MESSAGE_BUILDERS: dict[
    str,
    Callable[[str, PromptSettings], list[ChatCompletionMessageParam]],
] = {
    JOB_INTENTS: build_intent_job_messages,
    JOB_PRACTICAL: build_practical_job_messages,
    JOB_HOTELS: build_hotel_job_messages,
    JOB_ROOMS: build_room_job_messages,
}


def build_job_messages(
    job: str,
    spec_json: str,
    settings: Settings,
) -> list[ChatCompletionMessageParam]:
    """Builds the chat messages for one brief of a job."""
    return MESSAGE_BUILDERS[job](spec_json, settings.prompts)


def select_pending(
    briefs: pd.DataFrame,
    finished: set[int],
    limit: int | None,
) -> pd.DataFrame:
    """Keeps the briefs that still have no text, up to an optional limit."""
    pending = briefs[~briefs["id"].isin(finished)]
    return pending if limit is None else pending.head(limit)


async def process_item(item_id: int, spec_json: str, context: JobContext) -> None:
    """Generates and stores the text of one brief."""
    request = context.request
    messages = build_job_messages(request.job, spec_json, context.settings)
    extra_body = THINKING_OFF if request.disable_thinking else None
    async with context.semaphore:
        reply, attempts = await generate_valid_text(
            context.client,
            request.model,
            messages,
            context.diversity.wrap(
                pick_validator(request.job, context.validator, spec_json)
            ),
            context.settings.text.maximum_tokens[request.job],
            context.settings.text,
            extra_body,
        )
    if reply is not None:
        context.diversity.register(reply)
    record = TextRecord(id=item_id, text=reply, model=request.model, attempts=attempts)
    await context.writer.write(record)


def build_diversity_filter(
    existing: dict[int, str],
    settings: Settings,
) -> DiversityFilter:
    """Creates the diversity filter, aware of the texts already written."""
    required = collect_required_phrases(settings)
    exempt = [split_words(phrase) for phrase in required]
    diversity = DiversityFilter(settings.text, exempt)
    for value in existing.values():
        diversity.register(value)
    return diversity


def build_job_context(
    request: JobRequest,
    total: int,
    existing: dict[int, str],
    validator: TextValidator,
    settings: Settings,
) -> JobContext:
    """Prepares the client, the limits and the writer of a job."""
    text = settings.text
    return JobContext(
        request=request,
        client=build_client(text),
        semaphore=asyncio.Semaphore(text.concurrency),
        writer=JobWriter(request.output_path, total, text.progress_every),
        validator=validator,
        diversity=build_diversity_filter(existing, settings),
        settings=settings,
    )


def build_acceptance_check(
    job: str,
    briefs: pd.DataFrame,
    validator: TextValidator,
) -> Callable[[TextRecord], bool]:
    """Builds the rule that decides whether a stored record is still valid."""
    specs = dict(zip(briefs["id"].tolist(), briefs["spec"].tolist()))

    def accepts(record: TextRecord) -> bool:
        if record.text is None:
            return False
        return pick_validator(job, validator, str(specs[record.id]))(record.text)

    return accepts


def prepare_job(
    request: JobRequest,
    jobs: TextJobs,
    settings: Settings,
) -> tuple[JobContext, pd.DataFrame, Callable[[TextRecord], bool]]:
    """Cleans the stored file and prepares what a job still has to write."""
    briefs = jobs.get(request.job)
    validator = TextValidator(settings.text, collect_place_words(settings))
    accepts = build_acceptance_check(request.job, briefs, validator)
    compact_records(request.output_path, accepts)
    existing = read_text_map(request.output_path)
    pending = select_pending(briefs, set(existing), request.limit)
    context = build_job_context(request, len(pending), existing, validator, settings)
    return context, pending, accepts


async def run_job(request: JobRequest, jobs: TextJobs, settings: Settings) -> str:
    """Generates all pending texts of a job and returns a summary line."""
    context, pending, accepts = prepare_job(request, jobs, settings)
    tasks: list[Coroutine[None, None, None]] = []
    for item_id, spec_json in zip(pending["id"], pending["spec"]):
        tasks.append(process_item(int(item_id), str(spec_json), context))
    await asyncio.gather(*tasks)
    context.writer.close()
    compact_records(request.output_path, accepts)
    return context.writer.describe(len(pending))
