from fakes import route_by_prompt
from pseudolabel.labeler import parse_class


def test_describe_item_image_only_and_cached(labeler_with_fake, sofa_item):
    lab, vlm = labeler_with_fake(lambda c, g, j: '{"attributes": {"color": "серый"}}')
    assert lab.describe_item(sofa_item) == {"color": "серый"}
    lab.describe_item(sofa_item)
    assert len(vlm.calls) == 1
    parts = vlm.calls[0]["content"]
    assert sum(p["type"] == "image_url" for p in parts) == 1
    assert sofa_item.title not in " ".join(p.get("text", "") for p in parts)


def test_complete_params_no_call_when_full(labeler_with_fake, sofa_item_full_params):
    lab, vlm = labeler_with_fake(lambda c, g, j: "{}")
    assert lab.complete_params(sofa_item_full_params) == sofa_item_full_params.key_params
    assert vlm.calls == []


def test_complete_params_fills_only_missing_text_only(labeler_with_fake, sofa_item):
    lab, vlm = labeler_with_fake(lambda c, g, j: '{"params": {"forma": "угловой", "osnovnoy_cvet": "красный"}}')
    out = lab.complete_params(sofa_item)
    assert out["forma"] == "угловой" and out["osnovnoy_cvet"] == sofa_item.key_params["osnovnoy_cvet"]
    assert all(p["type"] == "text" for p in vlm.calls[0]["content"])
    lab.complete_params(sofa_item)
    assert len(vlm.calls) == 1


def test_label_pair_stores_vlm_label(labeler_with_fake, sofa_item, sofa_item_b, tmp_store):
    lab, vlm = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 4}'))
    r = lab.label_pair(sofa_item, sofa_item_b)
    assert r.cls == 4 and r.error is None and r.visual_a == {"color": "серый"}
    assert tmp_store.get_vlm_label("sofas", sofa_item.id, sofa_item_b.id, "fake-vlm", 1).cls == 4
    match_call = vlm.calls[-1]
    assert sum(p["type"] == "image_url" for p in match_call["content"]) == 2 and match_call["json_mode"] is False
    assert match_call["gen"]["thinking"] is True and match_call["gen"]["temperature"] == 0.1


def test_label_pair_cached(labeler_with_fake, sofa_item, sofa_item_b):
    lab, vlm = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 4}'))
    lab.label_pair(sofa_item, sofa_item_b)
    n = len(vlm.calls)
    assert lab.label_pair(sofa_item, sofa_item_b).cls == 4
    assert len(vlm.calls) == n


def test_parse_class_variants():
    assert parse_class({"class": 4}) == 4
    assert parse_class({"class": "4"}) == 4
    assert parse_class({"Класс": 2}) == 2
    assert parse_class({"class": 7}) is None
    assert parse_class({"class": "четыре"}) is None
    assert parse_class({}) is None


def test_label_pair_unparseable_class_records_error(labeler_with_fake, sofa_item, sofa_item_b):
    lab, _ = labeler_with_fake(route_by_prompt(match='{"reasoning": "hm", "class": 9}'))
    r = lab.label_pair(sofa_item, sofa_item_b)
    assert r.cls is None and "class" in r.error


def test_label_pair_missing_image_records_error(labeler_with_fake, sofa_item, missing_image_item, tmp_store):
    lab, _ = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 4}'))
    r = lab.label_pair(sofa_item, missing_image_item)
    assert r.cls is None and "image" in r.error.lower()
    assert tmp_store.query_labels("sofas", source="vlm").iloc[0]["error"]


def test_on_call_reports_steps(tmp_store, sofa_profile, sofa_item, sofa_item_b):
    from fakes import FakeVLM
    from pseudolabel.labeler import Labeler

    seen = []
    lab = Labeler(FakeVLM(route_by_prompt()), tmp_store, sofa_profile, 1, retry_sleep_s=0,
                  on_call=lambda step, status, latency: seen.append((step, status)))
    lab.label_pair(sofa_item, sofa_item_b)
    assert ("visual", "ok") in seen and ("match", "ok") in seen and ("text_params", "ok") in seen


def test_parse_class_floats_and_key_case():
    assert parse_class({"class": 4.0}) == 4
    assert parse_class({"class": "4.0"}) == 4
    assert parse_class({"Class": 3}) == 3
    assert parse_class({"class": 4.5}) is None


def test_json_mode_only_without_thinking(labeler_with_fake, sofa_profile, sofa_item, sofa_item_b):
    lab, vlm = labeler_with_fake(route_by_prompt(match='thinking {draft} </think> {"reasoning": "ok", "class": 4}'))
    assert lab.label_pair(sofa_item, sofa_item_b).cls == 4
    assert vlm.calls[-1]["json_mode"] is False
    sofa_profile.thinking = False
    lab2, vlm2 = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 4}'), profile=sofa_profile)
    lab2.label_pair(sofa_item, sofa_item_b, use_cache=False)
    assert vlm2.calls[-1]["json_mode"] is True


def _all_text(vlm) -> str:
    from fakes import prompt_text

    return " ".join(prompt_text(c["content"]) for c in vlm.calls)


def _markers(vlm) -> list[str]:
    from fakes import prompt_text

    return [prompt_text(c["content"]).split("]")[0] + "]" for c in vlm.calls]


def test_photo_only_sends_no_text(labeler_with_fake, sofa_item, sofa_item_b):
    from pseudolabel.labeler import LabelerOptions
    from pseudolabel.prompts import TEXT_MARKER

    lab, vlm = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 2}'), options=LabelerOptions(use_text=False))
    r = lab.label_pair(sofa_item, sofa_item_b)
    text = _all_text(vlm)
    assert r.cls == 2 and TEXT_MARKER not in text
    # Item b's description is a single common word that also appears in the attribute descriptions.
    assert sofa_item.title not in text and sofa_item_b.title not in text and sofa_item.description not in text
    assert "Parameters:" not in text and "Attributes seen in the photo" in text


def test_one_step_single_call(labeler_with_fake, sofa_item, sofa_item_b):
    from pseudolabel.labeler import LabelerOptions
    from pseudolabel.prompts import MATCH_MARKER

    lab, vlm = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 4}'),
                                 options=LabelerOptions(visual_step=False, text_step=False))
    r = lab.label_pair(sofa_item, sofa_item_b)
    assert r.cls == 4 and _markers(vlm) == [MATCH_MARKER]
    call = vlm.calls[0]
    assert sum(p["type"] == "image_url" for p in call["content"]) == 2
    text = _all_text(vlm)
    assert sofa_item.title in text and "Attributes seen in the photo" not in text
    assert r.params_a["forma"] is None and r.visual_a == {}


def test_no_visual_step_skips_visual_call(labeler_with_fake, sofa_item, sofa_item_b):
    from pseudolabel.labeler import LabelerOptions
    from pseudolabel.prompts import MATCH_MARKER, TEXT_MARKER

    lab, vlm = labeler_with_fake(route_by_prompt(text='{"params": {"forma": "угловой"}}'),
                                 options=LabelerOptions(visual_step=False))
    r = lab.label_pair(sofa_item, sofa_item_b)
    assert _markers(vlm) == [TEXT_MARKER, MATCH_MARKER] and r.params_a["forma"] == "угловой"


def test_no_text_step_keeps_missing_params_empty(labeler_with_fake, sofa_item, sofa_item_b):
    from pseudolabel.labeler import LabelerOptions
    from pseudolabel.prompts import TEXT_MARKER

    lab, vlm = labeler_with_fake(route_by_prompt(), options=LabelerOptions(text_step=False))
    r = lab.label_pair(sofa_item, sofa_item_b)
    assert TEXT_MARKER not in _all_text(vlm) and r.params_a["forma"] is None
    assert sofa_item.title in _all_text(vlm)


def test_few_shot_adds_example_images(labeler_with_fake, sofa_item, sofa_item_b, sofa_item_full_params):
    from pseudolabel.labeler import LabelerOptions

    opts = LabelerOptions(few_shot=[(sofa_item_full_params, sofa_item_b, 5)])
    lab, vlm = labeler_with_fake(route_by_prompt(match='{"reasoning": "ok", "class": 3}'), options=opts)
    assert lab.label_pair(sofa_item, sofa_item_b).cls == 3
    match = vlm.calls[-1]["content"]
    assert sum(p["type"] == "image_url" for p in match) == 4
    texts = [p["text"] for p in match if p["type"] == "text"]
    assert any("Correct class: 5" in t for t in texts) and "last two images" in texts[-1]
    assert match[-1]["type"] == "text" and match[-2]["type"] == "image_url" and match[-3]["type"] == "image_url"


def test_namespace_separates_label_cache(tmp_store, sofa_profile, sofa_item, sofa_item_b):
    from fakes import FakeVLM
    from pseudolabel.labeler import Labeler, LabelerOptions

    vlm = FakeVLM(route_by_prompt(match='{"reasoning": "ok", "class": 4}'))
    for ns in ("sofas@base", "sofas@base_repeat"):
        Labeler(vlm, tmp_store, sofa_profile, 1, retry_sleep_s=0, options=LabelerOptions(namespace=ns)).label_pair(sofa_item, sofa_item_b)
    assert sum(1 for c in vlm.calls if "[step: match]" in c["content"][-1]["text"]) == 2
    assert tmp_store.get_vlm_label("sofas@base_repeat", sofa_item.id, sofa_item_b.id, "fake-vlm", 1).cls == 4
    assert tmp_store.get_vlm_label("sofas", sofa_item.id, sofa_item_b.id, "fake-vlm", 1) is None


def test_probs_and_confidence_from_logprobs(labeler_with_fake, sofa_item, sofa_item_b):
    import math

    from fakes import prompt_text
    from pseudolabel.prompts import MATCH_MARKER
    from pseudolabel.vlm import Completion

    tokens = ['{"', "reasoning", '":', ' "ok",', ' "', "class", '":', " ", "4", "}"]
    lp = [{"token": t, "top_logprobs": [{"token": t, "logprob": 0.0}]} for t in tokens]
    lp[8]["top_logprobs"] = [{"token": "4", "logprob": math.log(0.6)}, {"token": "5", "logprob": math.log(0.4)}]
    base = route_by_prompt()

    def handler(content, gen, json_mode):
        if prompt_text(content).startswith(MATCH_MARKER):
            return Completion('{"reasoning": "ok", "class": 4}', lp, 11)
        return base(content, gen, json_mode)

    lab, vlm = labeler_with_fake(handler)
    r = lab.label_pair(sofa_item, sofa_item_b)
    assert r.cls == 4 and abs(r.probs[4] - 0.6) < 1e-9 and abs(r.confidence - 0.6) < 1e-9
    assert vlm.calls[-1]["top_logprobs"] == 5 and r.completion_tokens == 11
