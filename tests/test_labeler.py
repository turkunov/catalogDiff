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
