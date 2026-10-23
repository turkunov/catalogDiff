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
