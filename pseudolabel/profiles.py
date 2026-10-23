"""Category profiles: everything category-specific lives here, never in code."""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

STEPS = ("visual", "text_params", "match")


class ProfileError(ValueError):
    pass


@dataclass
class Attribute:
    key: str
    description: str


@dataclass
class SideInput:
    """Column names of one side of a pairs file; `image` is a str.format template over the row."""

    id: str
    title: str
    description: str
    key_params: str | None
    image: str
    category: str | None = None


@dataclass
class Example:
    name: str
    expected_class: int
    a: dict
    b: dict


@dataclass
class Profile:
    name: str
    category_label: str
    reasoning_language: str
    input_a: SideInput
    input_b: SideInput
    image_root: str
    visual_attributes: list[Attribute]
    key_params: list[Attribute]
    class_definitions: dict[int, str]
    match_rules: str
    generation: dict[str, dict]
    thinking: bool = True
    categories_a: list[str] = field(default_factory=list)
    categories_b: list[str] = field(default_factory=list)
    legacy_label_column: str | None = None
    prompt_overrides: dict[str, str | None] = field(default_factory=lambda: {s: None for s in STEPS})
    examples: list[Example] = field(default_factory=list)
    object_prompt: str | None = None  # English noun for text-prompted segmentation (SAM3), e.g. "sofa"


def _to_dict(p: Profile) -> dict[str, Any]:
    return {
        "name": p.name,
        "category_label": p.category_label,
        "reasoning_language": p.reasoning_language,
        "input": {"a": asdict(p.input_a), "b": asdict(p.input_b), "image_root": p.image_root},
        "categories": {"a": list(p.categories_a), "b": list(p.categories_b)},
        "legacy_label_column": p.legacy_label_column,
        "visual_attributes": [asdict(a) for a in p.visual_attributes],
        "key_params": [asdict(a) for a in p.key_params],
        "class_definitions": {int(k): v for k, v in sorted(p.class_definitions.items())},
        "match_rules": p.match_rules,
        "prompt_overrides": {s: p.prompt_overrides.get(s) for s in STEPS},
        "generation": {s: dict(p.generation.get(s, {})) for s in STEPS},
        "thinking": p.thinking,
        "examples": [asdict(e) for e in p.examples],
        "object_prompt": p.object_prompt,
    }


def _require(d: dict, key: str, ctx: str) -> Any:
    if key not in d or d[key] is None:
        raise ProfileError(f"{ctx}: missing '{key}'")
    return d[key]


def _attributes(items: Any, ctx: str) -> list[Attribute]:
    if not isinstance(items, list):
        raise ProfileError(f"{ctx}: must be a list")
    out, seen = [], set()
    for item in items:
        key = str(_require(item, "key", ctx)).strip()
        if not key:
            raise ProfileError(f"{ctx}: empty key")
        if key in seen:
            raise ProfileError(f"{ctx}: duplicate key '{key}'")
        seen.add(key)
        out.append(Attribute(key=key, description=str(item.get("description", ""))))
    return out


def _side(d: Any, ctx: str) -> SideInput:
    if not isinstance(d, dict):
        raise ProfileError(f"{ctx}: must be a mapping")
    return SideInput(
        id=_require(d, "id", ctx),
        title=_require(d, "title", ctx),
        description=_require(d, "description", ctx),
        key_params=d.get("key_params"),
        image=_require(d, "image", ctx),
        category=d.get("category"),
    )


def _from_dict(d: Any) -> Profile:
    if not isinstance(d, dict):
        raise ProfileError("profile must be a mapping")
    inp = _require(d, "input", "profile")
    raw_classes = _require(d, "class_definitions", "profile")
    try:
        classes = {int(k): str(v).strip() for k, v in raw_classes.items()}
    except (AttributeError, ValueError) as e:
        raise ProfileError(f"class_definitions: {e}") from e
    if sorted(classes) != [1, 2, 3, 4, 5] or not all(classes.values()):
        raise ProfileError("class_definitions: need non-empty texts for exactly classes 1..5")
    generation = d.get("generation") or {}
    match_gen = generation.get("match") or {}
    for key in ("temperature", "top_p", "max_tokens"):
        if key not in match_gen:
            raise ProfileError(f"generation.match: missing '{key}'")
    examples = []
    for e in d.get("examples") or []:
        cls = int(_require(e, "expected_class", "examples"))
        if cls not in range(1, 6):
            raise ProfileError(f"examples: expected_class {cls} not in 1..5")
        examples.append(Example(name=str(_require(e, "name", "examples")), expected_class=cls,
                                a=dict(_require(e, "a", "examples")), b=dict(_require(e, "b", "examples"))))
    overrides = d.get("prompt_overrides") or {}
    categories = d.get("categories") or {}
    return Profile(
        name=str(_require(d, "name", "profile")),
        category_label=str(_require(d, "category_label", "profile")),
        reasoning_language=str(d.get("reasoning_language", "en")),
        input_a=_side(_require(inp, "a", "input"), "input.a"),
        input_b=_side(_require(inp, "b", "input"), "input.b"),
        image_root=str(inp.get("image_root", "")),
        visual_attributes=_attributes(_require(d, "visual_attributes", "profile"), "visual_attributes"),
        key_params=_attributes(d.get("key_params") or [], "key_params"),
        class_definitions=classes,
        match_rules=str(d.get("match_rules", "")),
        generation={s: dict(generation.get(s) or {}) for s in STEPS},
        thinking=bool(d.get("thinking", True)),
        categories_a=[str(c) for c in categories.get("a") or []],
        categories_b=[str(c) for c in categories.get("b") or []],
        legacy_label_column=d.get("legacy_label_column"),
        prompt_overrides={s: overrides.get(s) for s in STEPS},
        examples=examples,
        object_prompt=d.get("object_prompt"),
    )


def parse_profile(yaml_text: str) -> Profile:
    from pseudolabel.prompts import check_overrides  # prompts imports this module

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ProfileError(f"invalid YAML: {e}") from e
    profile = _from_dict(data)
    check_overrides(profile)
    return profile


def load_profile(path: Path) -> Profile:
    return parse_profile(Path(path).read_text(encoding="utf-8"))


def dump_profile(p: Profile) -> str:
    return yaml.safe_dump(_to_dict(p), allow_unicode=True, sort_keys=False, width=100)


def profile_hash(p: Profile) -> str:
    canonical = json.dumps(_to_dict(p), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def list_profiles(profiles_dir: Path) -> list[str]:
    return sorted(path.stem for path in Path(profiles_dir).glob("*.yaml"))
