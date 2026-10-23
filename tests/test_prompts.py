import pytest

from pseudolabel.profiles import ProfileError
from pseudolabel.prompts import MATCH_MARKER, TEXT_MARKER, VISUAL_MARKER, match_prompt, text_params_prompt, visual_prompt


def test_visual_prompt_has_attribute_keys_only(sofa_profile):
    text = visual_prompt(sofa_profile)
    for k in ["color", "form", "upholstery", "covered"]:
        assert k in text
    assert "material_obivki" not in text
    assert text.startswith(VISUAL_MARKER) and '"attributes"' in text and "image_unavailable" in text


def test_text_params_prompt_only_missing(sofa_profile, sofa_item):
    text = text_params_prompt(sofa_profile, sofa_item, ["forma"])
    assert sofa_item.title in text and sofa_item.description in text
    assert "forma" in text and "osnovnoy_cvet" not in text
    assert text.startswith(TEXT_MARKER) and '"params"' in text


def test_match_prompt_contents(sofa_profile, sofa_item, sofa_item_b):
    text = match_prompt(sofa_profile, sofa_item, sofa_item_b, {"forma": "прямой"}, {}, {"color": "серый"}, {"color": "синий"})
    for i in range(1, 6):
        assert sofa_profile.class_definitions[i] in text
    assert sofa_item.title in text and sofa_item_b.title in text and "синий" in text
    assert sofa_profile.match_rules in text and "Russian" in text
    assert '"class"' in text and text.startswith(MATCH_MARKER)


def test_override_used(sofa_profile):
    sofa_profile.prompt_overrides["visual"] = "Describe {category}: {visual_attributes}"
    text = visual_prompt(sofa_profile)
    assert text.startswith("Describe Диваны:") and "upholstery" in text


def test_override_unknown_placeholder(sofa_profile):
    sofa_profile.prompt_overrides["visual"] = "Describe {nonexistent}"
    with pytest.raises(ProfileError):
        visual_prompt(sofa_profile)
