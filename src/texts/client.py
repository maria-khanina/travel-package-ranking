"""Requests texts from an OpenAI-compatible server such as vLLM."""

import asyncio
from collections.abc import Callable
import re

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.settings.records import TextGenerationSettings

QUOTE_CHARACTERS = "\"'“”"
LABEL_PATTERN = re.compile(
    r"^(message|description|room description)\s*:\s*",
    re.IGNORECASE,
)
ExtraBody = dict[str, dict[str, bool]]


def clean_generated_text(raw: str) -> str:
    """Strips labels and wrapping quotes that models sometimes add."""
    text = LABEL_PATTERN.sub("", raw.strip())
    return text.strip(QUOTE_CHARACTERS).strip()


def build_client(text: TextGenerationSettings) -> AsyncOpenAI:
    """Creates the asynchronous client for the configured server."""
    return AsyncOpenAI(
        base_url=text.base_url,
        api_key=text.api_key,
        timeout=text.request_timeout_seconds,
    )


async def request_text(
    client: AsyncOpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    maximum_tokens: int,
    text: TextGenerationSettings,
    extra_body: ExtraBody | None,
) -> str | None:
    """Sends one chat request and returns the cleaned reply, or None on failure."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=maximum_tokens,
            temperature=text.temperature,
            top_p=text.top_p,
            extra_body=extra_body,
        )
    except Exception:
        return None
    return clean_generated_text(response.choices[0].message.content or "")


async def generate_valid_text(
    client: AsyncOpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    validate: Callable[[str], bool],
    maximum_tokens: int,
    text: TextGenerationSettings,
    extra_body: ExtraBody | None,
) -> tuple[str | None, int]:
    """Retries a request until the reply passes validation or attempts run out."""
    for attempt in range(1, text.attempts + 1):
        reply = await request_text(
            client,
            model,
            messages,
            maximum_tokens,
            text,
            extra_body,
        )
        if reply is None:
            await asyncio.sleep(text.retry_backoff_seconds * attempt)
        elif validate(reply):
            return reply, attempt
    return None, text.attempts
