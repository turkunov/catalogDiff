import re
from pathlib import Path

import pytest

from pseudolabel.profiles import (
    ProfileError,
    dump_profile,
    list_profiles,
    load_profile,
    parse_profile,
    profile_hash,
)

PROFILES = Path(__file__).parents[1] / "profiles"


def test_load_all_shipped_profiles():
    assert set(list_profiles(PROFILES)) >= {"sofas", "hoodies", "cars"}
    for name in ["sofas", "hoodies", "cars"]:
        p = load_profile(PROFILES / f"{name}.yaml")
        assert sorted(p.class_definitions) == [1, 2, 3, 4, 5]


def test_sofas_prompt_fixes():
    p = load_profile(PROFILES / "sofas.yaml")
    rules = p.match_rules.lower()
    assert not re.search(r"игнорир\w*[^.]*цвет", rules)
    assert "гб" not in rules and "ма·ч" not in rules
    assert [a.key for a in p.key_params] == ["material_obivki", "forma", "osnovnoy_cvet"]


def test_hash_stable_and_sensitive():
    p = load_profile(PROFILES / "sofas.yaml")
    assert profile_hash(p) == profile_hash(parse_profile(dump_profile(p)))
    p.match_rules += " "
    assert profile_hash(p) != profile_hash(load_profile(PROFILES / "sofas.yaml"))


def test_validate_missing_class_definition():
    text = dump_profile(load_profile(PROFILES / "sofas.yaml")).replace("\n  5:", "\n  6:")
    with pytest.raises(ProfileError, match="class_definitions"):
        parse_profile(text)


def test_validate_duplicate_attribute():
    p = load_profile(PROFILES / "sofas.yaml")
    p.visual_attributes.append(p.visual_attributes[0])
    with pytest.raises(ProfileError, match="duplicate"):
        parse_profile(dump_profile(p))


def test_validate_broken_yaml():
    with pytest.raises(ProfileError):
        parse_profile("name: [unclosed")


def test_validate_broken_prompt_override():
    p = load_profile(PROFILES / "sofas.yaml")
    p.prompt_overrides["match"] = 'Return {"class": 1}'
    with pytest.raises(ProfileError, match="prompt_overrides.match"):
        parse_profile(dump_profile(p))


def test_object_prompt_round_trips():
    from pseudolabel.profiles import dump_profile, load_profile, parse_profile

    p = load_profile(PROFILES / "sofas.yaml")
    assert p.object_prompt == "sofa"
    assert parse_profile(dump_profile(p)).object_prompt == "sofa"
    assert load_profile(PROFILES / "cars.yaml").object_prompt == "car"


def test_object_prompt_does_not_change_the_profile_hash():
    """Profile versions registered before object_prompt existed keep their hash (Resume, label caches)."""
    import hashlib
    import json
    from dataclasses import replace

    from pseudolabel.profiles import _to_dict, load_profile, profile_hash

    p = load_profile(PROFILES / "sofas.yaml")
    old_format = {k: v for k, v in _to_dict(p).items() if k != "object_prompt"}
    old_hash = hashlib.sha256(json.dumps(old_format, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert profile_hash(p) == old_hash == profile_hash(replace(p, object_prompt=None))
