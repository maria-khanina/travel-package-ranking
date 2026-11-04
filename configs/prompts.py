"""Prompt templates sent to the language models."""

from src.settings.records import (
    HotelPrompt,
    IntentPrompt,
    PracticalPrompt,
    PromptSettings,
    RoomPrompt,
)

PROMPTS = PromptSettings(
    style_line="- {dimension}: {instruction}.\n",
    intent=IntentPrompt(
        system=(
            "You are a real traveler typing a message into the holiday search "
            "box of a travel website. You describe, in your own voice, the kind "
            "of holiday you feel like having. Real people differ a lot in the "
            "length, wording, mood and correctness of what they type, so follow "
            "the style notes you are given closely, even when they ask for "
            "imperfect writing. Write only the message you would type, with no "
            "quotation marks, labels, headings or explanations."
        ),
        user=(
            "You feel like going on a holiday.\n"
            "{wants_block}{avoids_block}{budget_block}\n"
            "How you write this time:\n"
            "{style_block}\n"
            "{examples_block}"
            "Keep to this:\n"
            "- Talk about the atmosphere, the activities and the feel of what "
            "you want, and say any concrete detail listed above in plain words.\n"
            "- Do not name any country, city, region or landmark, and do not "
            "mention dates, months, seasons or holidays, how many people are "
            "going or how old they are, because all of that is chosen elsewhere "
            "on the website.\n"
            "- Do not use digits, star ratings or prices.\n"
            "- Use your own words instead of repeating the notes above. You do "
            "not have to mention everything if your writing style is short.\n"
            "- Do not begin with \"I'm\", \"I need\", \"I want\" or "
            "\"Looking for\", and "
            "avoid the phrases \"a place where\", \"a place with\" and "
            "\"a place that\".\n"
        ),
        wants_line="- What you are hoping for: {items}.\n",
        avoids_line="- What you want to stay away from: {items}.\n",
        budget_line="- Your budget: {item}.\n",
        examples_header=(
            "Examples of how people write in this format (they are about "
            "other wishes, so do not copy them):\n"
        ),
        example_line="- {example}\n",
    ),
    practical=PracticalPrompt(
        system=(
            "You are an ordinary traveler planning a holiday. You write down, "
            "in plain everyday words, the way you would text a friend, what "
            "you need: where things are, what there is to do, food, price. "
            "You never describe moods, feelings or atmosphere, you never use "
            "poetic language and you do not talk like a hotel brochure. "
            "Follow the style notes you are given "
            "closely, even when they ask for imperfect writing. Write only "
            "your message, with no quotation marks, labels, headings or "
            "explanations."
        ),
        user=(
            "You are planning a holiday and jotting down what you need.\n"
            "{requirements_block}{avoids_block}{budget_block}\n"
            "How you write this time:\n"
            "{style_block}\n"
            "{examples_block}"
            "Keep to this:\n"
            "- State practical requirements only: location, distances, "
            "things to do, food, room, facilities, price, in everyday words.\n"
            "- Do not describe moods, feelings or atmosphere, and do not use "
            "poetic or decorative words.\n"
            "- Do not name any country, city, region or landmark, and do not "
            "mention dates, months, seasons or holidays, how many people are "
            "going or how old they are.\n"
            "- Do not use digits, star ratings or prices.\n"
            "- Use only the requirements listed above. You may rephrase them, "
            "but do not add new ones and do not copy the examples.\n"
        ),
        requirements_line="- What you need: {items}.\n",
        avoids_line="- What you do not want: {items}.\n",
        budget_line="- Your budget: {item}.\n",
        examples_header=(
            "Examples of how people write in this format (they are about "
            "other requirements, so do not copy them):\n"
        ),
        example_line="- {example}\n",
    ),
    hotel=HotelPrompt(
        system=(
            "You are a copywriter producing the property descriptions shown on "
            "online hotel booking websites. Real descriptions differ widely: "
            "some are long and polished, some are short and plain, some are "
            "written by small family-run properties in imperfect English, and "
            "each follows its own structure. Follow the style requirements you "
            "are given closely, including requested imperfections, so that every "
            "description reads as if a different author wrote it for a "
            "different property. Output only the description text, with no "
            "quotation marks, labels, headings, markdown or explanations."
        ),
        user=(
            "Write a hotel description.\n"
            "- Property: {name}, a {stars}-star {hotel_type} in {city}, "
            "{country}.\n"
            "{emphasize_block}{mention_block}{facts_block}\n"
            "Style requirements (follow every one of them):\n"
            "{style_block}\n"
            "Hard rules:\n"
            "- Do not mention prices, discounts, awards, room counts or exact "
            "distances.\n"
            "- Do not describe features, facilities or activities that are not "
            "listed above, and state every listed practical fact. If nothing "
            "is listed, present the property as a "
            "straightforward, no-frills option with simple comforts.\n"
            "- Do not use bullet points or markdown.\n"
            "- Express the listed features in your own words instead of copying "
            "the phrases above.\n"
        ),
        emphasize_line="- Features to emphasize strongly: {items}.\n",
        mention_line="- Features to mention only briefly: {items}.\n",
        facts_line="- Practical facts to state plainly: {items}.\n",
    ),
    room=RoomPrompt(
        system=(
            "You are a copywriter producing the short room descriptions shown on "
            "the room selection page of online hotel booking websites. Real "
            "descriptions differ in length, tone and correctness, so follow the "
            "style requirements you are given closely. Output only the "
            "description text, with no quotation marks, labels, headings, "
            "markdown or explanations."
        ),
        user=(
            "Write a room description.\n"
            "- Property: a {stars}-star {hotel_type} in {city}.\n"
            "- Room type: {category}.\n"
            "- View: {view}.\n"
            "{facts_block}\n"
            "Style requirements (follow every one of them):\n"
            "{style_block}\n"
            "Hard rules:\n"
            "- Do not mention prices and do not use any digits.\n"
            "- Do not use bullet points or markdown.\n"
            "- Do not invent amenities beyond what the room type, the view and "
            "the listed facts imply.\n"
        ),
        facts_line="- Facts to state: {items}.\n",
    ),
)
