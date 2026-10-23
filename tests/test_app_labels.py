from pseudolabel.app.tab_labels import export_selection, pair_card, search_labels


def test_search_labels_filter_by_source(app_ctx_with_iteration):
    df = search_labels(app_ctx_with_iteration, "sofas", {"source": "vlm"})
    assert len(df) > 0 and set(df.source) == {"vlm"}
    assert len(search_labels(app_ctx_with_iteration, "sofas", {"source": "vlm", "cls": "", "origin": None})) == len(df)


def test_pair_card_and_export(app_ctx_with_iteration):
    df = search_labels(app_ctx_with_iteration, "sofas", {"source": "vlm"})
    row = df.iloc[0]
    card = pair_card(app_ctx_with_iteration, "sofas", row.a_id, row.b_id)
    assert card["a"]["id"] == row.a_id and card["labels"] and card["origins"]
    assert any(a["kind"] == "visual" for a in card["annotations"]["a"])
    assert export_selection(app_ctx_with_iteration, df).endswith(".csv")


def _with_confidences(ctx):
    from pseudolabel.store import LabelRow

    for b_id, conf in (("lc-1", 0.45), ("lc-2", 0.95)):
        ctx.store.put_label(LabelRow("sofas", "a-lc", b_id, "vlm", 4, model="fake-vlm", profile_version=1, confidence=conf))
    return ctx


def test_low_confidence_filter_and_flag(app_ctx_with_iteration):
    ctx = _with_confidences(app_ctx_with_iteration)
    df = search_labels(ctx, "sofas", {"source": "vlm", "low_confidence_below": 0.7})
    assert df["b_id"].tolist() == ["lc-1"] and df["low_confidence"].all()
    everything = search_labels(ctx, "sofas", {"source": "vlm", "confidence_threshold": 0.7})
    assert everything.set_index("b_id").loc["lc-2", "low_confidence"] == False  # noqa: E712


def test_send_low_confidence_to_manual_review(app_ctx_with_iteration):
    from pseudolabel.app.tab_labels import queue_low_confidence

    ctx = _with_confidences(app_ctx_with_iteration)
    df = search_labels(ctx, "sofas", {"source": "vlm", "confidence_threshold": 0.7})
    assert queue_low_confidence(ctx, "sofas", df) == 1
    assert ctx.store.queue_pairs("low_confidence:sofas") == [("a-lc", "lc-1")]
    assert "low_confidence:sofas" in ctx.store.queues("sofas")
