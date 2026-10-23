"""Prompts for the three labeler steps, built from a profile.

A profile may override any step with its own str.format template; the available fields are listed
in FIELDS. Default templates start with a marker line so logs and tests can tell the steps apart.
"""
import json

from pseudolabel.data import Item
from pseudolabel.profiles import Attribute, Profile, ProfileError

VISUAL_MARKER = "[step: visual attributes]"
TEXT_MARKER = "[step: parameters from text]"
MATCH_MARKER = "[step: match]"

FIELDS = (
    "category", "visual_attributes", "key_params", "class_definitions", "rules", "language", "missing_keys",
    "title", "description", "a_title", "a_description", "b_title", "b_description",
    "a_params", "b_params", "a_visual", "b_visual", "images_note", "listings",
)

MAX_DESCRIPTION_CHARS = 1500
LANGUAGES = {"ru": "Russian", "en": "English"}

VISUAL_TEMPLATE = VISUAL_MARKER + """
You see one photo from a listing in the category "{category}". Describe only what is visible in the photo.
Ignore people, background and unrelated objects.
Attributes to fill:
{visual_attributes}
Return only a JSON object, with no text before or after it:
{{"attributes": {{"<attribute key>": "<value or null>"}}}}
Use null when the photo does not show an attribute. Write values in {language}.
If the photo cannot be read or shows no item at all, return exactly {{"error": "image_unavailable"}}."""

TEXT_TEMPLATE = TEXT_MARKER + """
Extract product parameters from the text of a listing in the category "{category}".
Title: {title}
Description: {description}
Parameters to extract:
{key_params}
Return only a JSON object, with no text before or after it:
{{"params": {{"<parameter key>": "<value or null>"}}}}
Use only the keys listed above. Use null when the text does not state a parameter; do not guess. Write values in {language}."""

IMAGES_NOTE = "The first image belongs to listing 1, the second image to listing 2."
FEW_SHOT_IMAGES_NOTE = ("The solved examples come first; the last two images are the pair to label: "
                        "the first of them belongs to listing 1, the second to listing 2.")

MATCH_TEMPLATE = MATCH_MARKER + """
You compare two listings in the category "{category}" and decide whether they show the same product.
{images_note}

Classes:
{class_definitions}

Rules:
{rules}

{listings}

Check the classes in order from 1 to 5 and choose the lowest class whose description fits the pair.
Return only a JSON object, with no text before or after it:
{{"reasoning": "<short reasoning in {language}: which checks passed or failed>", "class": <integer 1-5>}}"""


def _attributes(attrs: list[Attribute]) -> str:
    return "\n".join(f"- {a.key}: {a.description}" for a in attrs)


def _json(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


def _desc(text: str) -> str:
    return text if len(text) <= MAX_DESCRIPTION_CHARS else text[:MAX_DESCRIPTION_CHARS] + "…"


def _base_fields(p: Profile) -> dict[str, str]:
    fields = dict.fromkeys(FIELDS, "")
    fields.update(
        category=p.category_label,
        visual_attributes=_attributes(p.visual_attributes),
        key_params=_attributes(p.key_params),
        class_definitions="\n".join(f"{k} — {v}" for k, v in sorted(p.class_definitions.items())),
        rules=p.match_rules,
        language=LANGUAGES.get(p.reasoning_language, p.reasoning_language),
    )
    return fields


def _render(p: Profile, step: str, default: str, fields: dict[str, str]) -> str:
    template = p.prompt_overrides.get(step) or default
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError) as e:
        raise ProfileError(f"prompt_overrides.{step}: bad placeholder {e}") from e


def check_overrides(p: Profile) -> None:
    """Render every overridden template with placeholder values; raises ProfileError on a bad template."""
    fields = dict.fromkeys(FIELDS, "x")
    for step, template in p.prompt_overrides.items():
        if template:
            _render(p, step, template, fields)


def visual_prompt(p: Profile) -> str:
    return _render(p, "visual", VISUAL_TEMPLATE, _base_fields(p))


def text_params_prompt(p: Profile, item: Item, missing: list[str]) -> str:
    fields = _base_fields(p)
    fields.update(
        key_params=_attributes([a for a in p.key_params if a.key in missing]),
        missing_keys=", ".join(missing),
        title=item.title,
        description=_desc(item.description),
    )
    return _render(p, "text_params", TEXT_TEMPLATE, fields)


def _listing(n: int, item: Item, params: dict, visual: dict, use_text: bool, show_visual: bool) -> str:
    lines = [f"Listing {n}"]
    if use_text:
        lines += [f"Title: {item.title}", f"Description: {_desc(item.description)}", f"Parameters: {_json(params)}"]
    if show_visual:
        lines.append(f"Attributes seen in the photo: {_json(visual)}")
    if len(lines) == 1:
        lines.append("Only the photo.")
    return "\n".join(lines)


def match_prompt(p: Profile, a: Item, b: Item, params_a: dict, params_b: dict, visual_a: dict, visual_b: dict,
                 use_text: bool = True, show_visual: bool = True, few_shot: bool = False) -> str:
    """Match step; use_text=False leaves out titles, descriptions and parameters, show_visual=False the photo attributes."""
    fields = _base_fields(p)
    fields.update(
        a_title=a.title, a_description=_desc(a.description), a_params=_json(params_a), a_visual=_json(visual_a),
        b_title=b.title, b_description=_desc(b.description), b_params=_json(params_b), b_visual=_json(visual_b),
        images_note=FEW_SHOT_IMAGES_NOTE if few_shot else IMAGES_NOTE,
        listings=_listing(1, a, params_a, visual_a, use_text, show_visual) + "\n\n"
        + _listing(2, b, params_b, visual_b, use_text, show_visual),
    )
    return _render(p, "match", MATCH_TEMPLATE, fields)


def few_shot_text(n: int, a: Item, b: Item, cls: int, use_text: bool = True) -> str:
    """Caption after the two images of solved example n."""
    head = f"Solved example {n} (not the pair to label): first image — listing 1, second image — listing 2."
    titles = f"\nListing 1 title: {a.title}\nListing 2 title: {b.title}" if use_text else ""
    return f"{head}{titles}\nCorrect class: {cls}"
