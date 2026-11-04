"""Style dimensions that make generated texts diverse."""

from src.settings.records import FormatOption, StyleOption, TextStyleSettings


TEXT_STYLES = TextStyleSettings(
    intent_formats=[
        FormatOption(
            label="keywords",
            probability=0.3,
            instruction=(
                "between three and six short keywords or "
                "fragments separated by commas, at most 12 "
                "words in total, no full sentences"
            ),
            maximum_words=14,
            skipped_dimensions=["opening", "persona"],
            examples=[
                "sunny beaches, lots of water, family-friendly vibe",
                "quiet, cozy, fireplace, snow outside",
                "party vibes, cheap drinks, beach all day",
                "old streets, good food, museums everywhere",
                "spa, silence, absolutely nothing to do",
                "adventure!! hiking, views, no crowds",
                "romantic dinners, sea view, slow mornings",
                "lazy resort, pool, kids club",
                "foodie city, wine, wandering around",
                "luxury, ocean view, spa",
                "chill + sunsets + good coffee",
                "mountains, lakes, fresh air, easy hikes",
                "buzzing nightlife, rooftop bars, beach nearby",
                "snow, hot chocolate, cozy cabin feel",
            ],
        ),
        FormatOption(
            label="fragment",
            probability=0.25,
            instruction=(
                "one short informal fragment of three to "
                "eight words, like a search query"
            ),
            maximum_words=14,
            skipped_dimensions=["opening"],
            examples=[
                "somewhere warm with a pool",
                "ski trip with good apres",
                "calm beach, no crowds",
                "city break with great food",
                "resort the kids will love",
                "wellness retreat vibes",
                "cheap and cheerful beach holiday",
                "cozy mountain cabin feel",
                "big nightlife, small budget",
                "snorkeling and sunsets",
                "peaceful countryside escape",
                "fun in the sun for the whole family",
                "something romantic and a bit fancy",
                "long lazy days by the water",
            ],
        ),
        FormatOption(
            label="quick_slang",
            probability=0.15,
            instruction=(
                "one or two short sloppy sentences typed fast "
                "on a phone, at most 15 words in total, slang "
                "and abbreviations are fine"
            ),
            maximum_words=22,
            skipped_dimensions=["opening"],
            examples=[
                "tbh just want a beach and a cold drink lol",
                "need chill vibes asap, no party crowd pls",
                "something fun for the kids but ok for me too haha",
                "gimme snow and hot chocolate",
                "wanna party all night, sleep by the pool all day",
                "anywhere warm w good food ngl",
                "not too fancy, just relaxing",
                "big city energy, shopping + food",
                "so over work, need spa and silence",
                "kids go wild, i nap by the sea",
                "cheap n sunny pls",
                "mountains. snow. done.",
            ],
        ),
        FormatOption(
            label="casual_sentence",
            probability=0.15,
            instruction=(
                "one casual sentence of at most 25 words"
            ),
            maximum_words=28,
            skipped_dimensions=[],
            examples=[
                (
                    "I'd love something relaxing by the water with plenty for "
                    "the kids."
                ),
                (
                    "Something quiet and romantic where I can switch off for "
                    "a few days."
                ),
                "Hiking all day and eating well at night sounds perfect.",
                "A lively spot with great food and loads of nightlife.",
                "Somewhere cozy with snow outside and a fire inside.",
                "A proper spa break, no noise and no schedule.",
                (
                    "Beach, pool and a bit of culture on the side would be "
                    "great."
                ),
                "A fancy resort where everything is taken care of.",
                "Just sun, sea and good seafood, nothing complicated.",
                (
                    "A buzzing city with museums, markets and late-night "
                    "food."
                ),
            ],
        ),
        FormatOption(
            label="chatty",
            probability=0.1,
            instruction=(
                "two or three chatty sentences, about 30 to "
                "50 words in total"
            ),
            maximum_words=60,
            skipped_dimensions=[],
            examples=[
                (
                    "Honestly I just want to sit by the sea and do nothing. "
                    "Maybe a nice dinner now and then, but mostly quiet."
                ),
                (
                    "The kids want water slides and I want a hammock. "
                    "Somewhere with something for everyone."
                ),
                (
                    "I'm tired of crowds. Give me mountains, fresh air and a "
                    "cozy spot to come back to."
                ),
                (
                    "Nightlife and sun, honestly the more energy the better. "
                    "Sleep can wait."
                ),
                (
                    "Something romantic but not stuffy, with good food and a "
                    "view worth waking up for."
                ),
                (
                    "I love wandering old streets and trying everything on "
                    "the menu. A hotel with some character would be nice."
                ),
            ],
        ),
        FormatOption(
            label="detailed",
            probability=0.05,
            instruction=(
                "three to four sentences that explain what "
                "matters and why, about 60 to 80 words in "
                "total"
            ),
            maximum_words=95,
            skipped_dimensions=[],
            examples=[
                (
                    "I want to properly switch off this time. Quiet "
                    "surroundings, a spa I can spend hours in, good simple "
                    "food and no noise at night. Nothing flashy, just calm "
                    "and comfortable."
                ),
                (
                    "Big on active days: hiking, biking, being outside from "
                    "morning till dark. In the evening I want somewhere cozy "
                    "with hearty food and a bed that actually helps me "
                    "recover."
                ),
                (
                    "Looking for a proper family holiday: sandy beach, warm "
                    "shallow water, a kids club and things to do on rainy "
                    "days. The grown-ups need a bit of calm and decent coffee "
                    "too."
                ),
            ],
        ),
    ],
    intent={
        "persona": [
            StyleOption(
                label="regular",
                probability=0.3,
                instruction="an ordinary traveler with no particular quirks",
            ),
            StyleOption(
                label="first_timer",
                probability=0.1,
                instruction=(
                    "someone who rarely travels and is a little unsure what they "
                    "want"
                ),
            ),
            StyleOption(
                label="experienced",
                probability=0.15,
                instruction=(
                    "an experienced traveler who knows exactly what they like and "
                    "is direct about it"
                ),
            ),
            StyleOption(
                label="indecisive",
                probability=0.1,
                instruction="someone torn between a couple of ideas who admits it",
            ),
            StyleOption(
                label="enthusiastic",
                probability=0.15,
                instruction="an excited person who uses expressive, upbeat language",
            ),
            StyleOption(
                label="practical",
                probability=0.2,
                instruction="a practical, to-the-point person who dislikes fluff",
            ),
        ],
        "imperfections": [
            StyleOption(
                label="clean",
                probability=0.65,
                instruction="correct spelling and normal punctuation",
            ),
            StyleOption(
                label="typos",
                probability=0.12,
                instruction="contains one or two natural typos or missing letters",
            ),
            StyleOption(
                label="lowercase",
                probability=0.1,
                instruction=(
                    "written entirely in lowercase with very little punctuation"
                ),
            ),
            StyleOption(
                label="slang",
                probability=0.08,
                instruction=(
                    "uses casual slang and abbreviations such as tbh, pls or lol"
                ),
            ),
            StyleOption(
                label="run_on",
                probability=0.05,
                instruction="one run-on sentence with missing punctuation",
            ),
        ],
        "opening": [
            StyleOption(
                label="free",
                probability=0.35,
                instruction="begin however feels natural",
            ),
            StyleOption(
                label="place_first",
                probability=0.25,
                instruction="begin directly with the kind of place wanted",
            ),
            StyleOption(
                label="mood_first",
                probability=0.15,
                instruction="begin with a feeling or mood",
            ),
            StyleOption(
                label="question",
                probability=0.1,
                instruction="begin with a question",
            ),
            StyleOption(
                label="verb_first",
                probability=0.15,
                instruction=(
                    "begin with any one of these words: Need, Want, Hoping for, "
                    "Seeking, Picturing, Longing for, Wishing for, Planning, "
                    "Fancying, Itching for"
                ),
            ),
        ],
    },
    hotel={
        "tone": [
            StyleOption(
                label="warm",
                probability=0.2,
                instruction="warm and welcoming, like a friendly host",
            ),
            StyleOption(
                label="refined",
                probability=0.15,
                instruction="refined and elegant, with understated sophistication",
            ),
            StyleOption(
                label="modern",
                probability=0.2,
                instruction="modern and casual, short punchy phrases",
            ),
            StyleOption(
                label="formal",
                probability=0.1,
                instruction="formal and corporate, like a hotel group brochure",
            ),
            StyleOption(
                label="promotional",
                probability=0.15,
                instruction="enthusiastic and promotional, selling the stay hard",
            ),
            StyleOption(
                label="plain",
                probability=0.2,
                instruction="plain and factual, with little adjectival decoration",
            ),
        ],
        "length": [
            StyleOption(
                label="short",
                probability=0.25,
                instruction="about 30 to 50 words in two sentences",
            ),
            StyleOption(
                label="medium",
                probability=0.5,
                instruction="about 60 to 100 words in three or four sentences",
            ),
            StyleOption(
                label="long",
                probability=0.25,
                instruction="about 110 to 150 words in five or six sentences",
            ),
        ],
        "structure": [
            StyleOption(
                label="paragraph",
                probability=0.35,
                instruction="a single flowing paragraph",
            ),
            StyleOption(
                label="location_first",
                probability=0.2,
                instruction=(
                    "open with the location and surroundings, then the facilities"
                ),
            ),
            StyleOption(
                label="feature_first",
                probability=0.2,
                instruction="open with the most striking feature, then the location",
            ),
            StyleOption(
                label="guest_focus",
                probability=0.15,
                instruction="address the guest directly using you and your",
            ),
            StyleOption(
                label="enumerating",
                probability=0.1,
                instruction=(
                    "run through the facilities in one long sentence separated by "
                    "commas"
                ),
            ),
        ],
        "imperfections": [
            StyleOption(
                label="clean",
                probability=0.94,
                instruction="correct spelling and grammar",
            ),
            StyleOption(
                label="translated",
                probability=0.03,
                instruction=(
                    "slightly awkward English, as if a small family-run property "
                    "translated its own text"
                ),
            ),
            StyleOption(
                label="hype",
                probability=0.02,
                instruction=(
                    "uses exclamation marks and occasional capital letters for "
                    "emphasis"
                ),
            ),
            StyleOption(
                label="typos",
                probability=0.01,
                instruction="contains one or two small typos or missing articles",
            ),
        ],
    },
    room={
        "length": [
            StyleOption(
                label="one_sentence",
                probability=0.35,
                instruction="one short sentence of 10 to 20 words",
            ),
            StyleOption(
                label="two_sentences",
                probability=0.45,
                instruction="two sentences of 25 to 45 words in total",
            ),
            StyleOption(
                label="three_sentences",
                probability=0.2,
                instruction="three sentences of 50 to 70 words in total",
            ),
        ],
        "tone": [
            StyleOption(
                label="factual",
                probability=0.5,
                instruction="concrete and factual",
            ),
            StyleOption(
                label="inviting",
                probability=0.3,
                instruction="inviting and pleasant",
            ),
            StyleOption(
                label="telegraphic",
                probability=0.2,
                instruction="clipped and telegraphic, like an inventory entry",
            ),
        ],
        "focus": [
            StyleOption(
                label="view",
                probability=0.3,
                instruction="put the emphasis on the view",
            ),
            StyleOption(
                label="space",
                probability=0.25,
                instruction="put the emphasis on the space and furnishings",
            ),
            StyleOption(
                label="comfort",
                probability=0.25,
                instruction="put the emphasis on comfort and sleeping",
            ),
            StyleOption(
                label="balanced",
                probability=0.2,
                instruction="give equal weight to the view and the room itself",
            ),
        ],
        "imperfections": [
            StyleOption(
                label="clean",
                probability=0.96,
                instruction="correct spelling and grammar",
            ),
            StyleOption(
                label="typos",
                probability=0.02,
                instruction="contains one small typo",
            ),
            StyleOption(
                label="translated",
                probability=0.02,
                instruction=(
                    "slightly awkward English, as if translated from another "
                    "language"
                ),
            ),
        ],
    },
)
