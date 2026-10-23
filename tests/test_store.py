from pseudolabel.data import Item
from pseudolabel.store import LabelRow


def test_profile_versions(tmp_store, sofa_profile):
    assert tmp_store.register_profile_version(sofa_profile) == 1
    assert tmp_store.register_profile_version(sofa_profile) == 1
    sofa_profile.match_rules += " v2"
    assert tmp_store.register_profile_version(sofa_profile) == 2
    assert [v for v, _ in tmp_store.profile_versions("sofas")] == [1, 2]


def test_items_roundtrip(tmp_store):
    item = Item("1", "a", "диван", "описание", {"forma": None}, "/img/1.jpeg")
    tmp_store.upsert_items("sofas", [item])
    assert tmp_store.get_item("sofas", "a", "1") == item
    assert tmp_store.get_item("sofas", "b", "1") is None


def test_annotation_cache_ignores_errors(tmp_store):
    tmp_store.put_annotation("sofas", "a", "1", "visual", "m", 1, result={}, raw="", error="boom", latency_s=1.0)
    assert tmp_store.get_annotation("sofas", "a", "1", "visual", "m", 1) is None
    tmp_store.put_annotation("sofas", "a", "1", "visual", "m", 1, result={"color": "серый"}, raw="{}", error=None, latency_s=1.0)
    assert tmp_store.get_annotation("sofas", "a", "1", "visual", "m", 1) == {"color": "серый"}
    assert tmp_store.get_annotation("sofas", "a", "1", "visual", "m", 2) is None


def test_vlm_label_cache_keyed_by_model_and_version(tmp_store):
    tmp_store.put_label(LabelRow("sofas", "1", "2", "vlm", 4, model="m", profile_version=1))
    assert tmp_store.get_vlm_label("sofas", "1", "2", "m", 1).cls == 4
    assert tmp_store.get_vlm_label("sofas", "1", "2", "m", 2) is None


def test_vlm_label_with_error_is_not_cache(tmp_store):
    tmp_store.put_label(LabelRow("sofas", "1", "2", "vlm", None, error="timeout", model="m", profile_version=1))
    assert tmp_store.get_vlm_label("sofas", "1", "2", "m", 1) is None


def test_human_label_latest_wins_and_never_overwritten_by_vlm(tmp_store):
    tmp_store.put_label(LabelRow("sofas", "1", "2", "human", 3))
    tmp_store.put_label(LabelRow("sofas", "1", "2", "human", 5))
    tmp_store.put_label(LabelRow("sofas", "1", "2", "vlm", 1, model="m", profile_version=1))
    assert tmp_store.latest_human_label("sofas", "1", "2").cls == 5


def test_iteration_pairs_status(tmp_store):
    it = tmp_store.create_iteration("sofas", 1, {"budget": 2}, "fake@1")
    tmp_store.add_iteration_pairs(it, [dict(a_id="1", b_id="2", origin="top_1_10", rank=1, cosine=0.9, selected_for_vlm=True)])
    assert len(tmp_store.iteration_pairs(it, "pending")) == 1
    tmp_store.set_iteration_pair_status(it, "1", "2", "done")
    assert len(tmp_store.iteration_pairs(it, "pending")) == 0
    tmp_store.finish_iteration(it, "done", {"x": 1})
    assert tmp_store.get_iteration(it)["report"] == {"x": 1}
    assert list(tmp_store.list_iterations("sofas")["id"]) == [it]


def test_import_legacy_idempotent(tmp_store):
    assert tmp_store.import_legacy("sofas", {("1", "2"): 5}) == 1
    assert tmp_store.import_legacy("sofas", {("1", "2"): 5}) == 0


def test_manual_test_and_queues(tmp_store):
    tmp_store.set_manual_test("sofas", [("1", "2", 5), ("3", "4", 1)])
    tmp_store.set_manual_test("sofas", [("1", "2", 5)])
    assert tmp_store.manual_test_pairs("sofas") == [("1", "2", 5)]
    tmp_store.queue_put("disagreements:7", "sofas", [("1", "2")])
    assert tmp_store.queue_pairs("disagreements:7") == [("1", "2")]
    assert tmp_store.queues("sofas") == ["disagreements:7"]


def test_query_labels_joins_origin_and_filters(tmp_store):
    it = tmp_store.create_iteration("sofas", 1, {}, "fake@1")
    tmp_store.add_iteration_pairs(it, [
        dict(a_id="1", b_id="2", origin="top_1_10", rank=3, cosine=0.8, selected_for_vlm=True),
        dict(a_id="1", b_id="3", origin="top_11_100", rank=40, cosine=0.5, selected_for_vlm=True),
    ])
    tmp_store.put_label(LabelRow("sofas", "1", "2", "vlm", 2, model="m", profile_version=1))
    tmp_store.put_label(LabelRow("sofas", "1", "3", "vlm", 4, model="m", profile_version=1))
    tmp_store.queue_put(f"disagreements:{it}", "sofas", [("1", "2")])
    df = tmp_store.query_labels("sofas", source="vlm")
    assert set(df.columns) >= {"a_id", "b_id", "source", "cls", "origin", "rank", "cosine", "iteration_id", "error"}
    assert df.set_index("b_id").loc["2", "origin"] == "top_1_10"
    assert list(tmp_store.query_labels("sofas", origin="top_11_100").b_id) == ["3"]
    assert list(tmp_store.query_labels("sofas", disagreement=True).b_id) == ["2"]
