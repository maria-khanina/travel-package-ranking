"""Builds the chat messages that ask a language model to write the texts."""

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.settings.records import (
    HotelPrompt,
    IntentPrompt,
    PracticalPrompt,
    PromptSettings,
    RoomPrompt,
)
from src.texts.models import HotelSpec, IntentSpec, PracticalSpec, RoomSpec

ITEM_SEPARATOR = "; "


def wrap_messages(system: str, user: str) -> list[ChatCompletionMessageParam]:
    """Wraps a system prompt and a user prompt into chat messages."""
    return [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(role="user", content=user),
    ]


def render_items(template: str, items: list[str]) -> str:
    """Renders a bullet line for a non-empty list, otherwise nothing."""
    if not items:
        return ""
    return template.format(items=ITEM_SEPARATOR.join(items))


def render_styles(styles: dict[str, str], line: str) -> str:
    """Renders one requirement line for every style dimension."""
    text = ""
    for dimension, instruction in styles.items():
        label = dimension.replace("_", " ").capitalize()
        text += line.format(dimension=label, instruction=instruction)
    return text


def render_examples(examples: list[str], header: str, line: str) -> str:
    """Renders the example messages block, or nothing without examples."""
    if not examples:
        return ""
    lines = ""
    for example in examples:
        lines += line.format(example=example)
    return header + lines + "\n"


def build_intent_messages(
    spec: IntentSpec,
    prompt: IntentPrompt,
    prompts: PromptSettings,
) -> list[ChatCompletionMessageParam]:
    """Asks for a traveler message describing the vibe of a holiday."""
    budget = prompt.budget_line.format(item=spec.budget) if spec.budget else ""
    user = prompt.user.format(
        wants_block=render_items(prompt.wants_line, spec.wants),
        avoids_block=render_items(prompt.avoids_line, spec.avoids),
        budget_block=budget,
        style_block=render_styles(spec.styles, prompts.style_line),
        examples_block=render_examples(
            spec.examples,
            prompt.examples_header,
            prompt.example_line,
        ),
    )
    return wrap_messages(prompt.system, user)


def build_practical_messages(
    spec: PracticalSpec,
    prompt: PracticalPrompt,
    prompts: PromptSettings,
) -> list[ChatCompletionMessageParam]:
    """Asks for a plain list of practical requirements about a hotel."""
    budget = prompt.budget_line.format(item=spec.budget) if spec.budget else ""
    user = prompt.user.format(
        requirements_block=render_items(prompt.requirements_line, spec.requirements),
        avoids_block=render_items(prompt.avoids_line, spec.avoids),
        budget_block=budget,
        style_block=render_styles(spec.styles, prompts.style_line),
        examples_block=render_examples(
            spec.examples,
            prompt.examples_header,
            prompt.example_line,
        ),
    )
    return wrap_messages(prompt.system, user)


def build_hotel_messages(
    spec: HotelSpec,
    prompt: HotelPrompt,
    prompts: PromptSettings,
) -> list[ChatCompletionMessageParam]:
    """Asks for a marketing description of a hotel."""
    user = prompt.user.format(
        name=spec.name,
        stars=spec.stars,
        hotel_type=spec.hotel_type,
        city=spec.city,
        country=spec.country,
        emphasize_block=render_items(prompt.emphasize_line, spec.emphasize),
        mention_block=render_items(prompt.mention_line, spec.mention),
        facts_block=render_items(prompt.facts_line, spec.facts),
        style_block=render_styles(spec.styles, prompts.style_line),
    )
    return wrap_messages(prompt.system, user)


def build_room_messages(
    spec: RoomSpec,
    prompt: RoomPrompt,
    prompts: PromptSettings,
) -> list[ChatCompletionMessageParam]:
    """Asks for a short description of a hotel room."""
    user = prompt.user.format(
        stars=spec.stars,
        hotel_type=spec.hotel_type,
        city=spec.city,
        category=spec.category,
        view=spec.view,
        facts_block=render_items(prompt.facts_line, spec.facts),
        style_block=render_styles(spec.styles, prompts.style_line),
    )
    return wrap_messages(prompt.system, user)
