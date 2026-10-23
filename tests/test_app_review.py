import dataclasses

from pseudolabel.app.tab_review import import_csv, next_card, prepare_manual_test, submit


def test_review_card_is_blind(app_ctx_with_iteration):
    card = next_card(app_ctx_with_iteration, app_ctx_with_iteration.disagreement_queue)
    fields = set(dataclasses.asdict(card))
    assert not fields & {"cls", "reasoning", "origin", "rank", "cosine", "class"}
    assert card.a_title and card.position.startswith("1/")


def test_submit_advances_queue(app_ctx_with_iteration):
    ctx = app_ctx_with_iteration
    q = ctx.disagreement_queue
    total = len(ctx.store.queue_pairs(q))
    c1 = next_card(ctx, q)
    c2 = submit(ctx, q, c1.a_id, c1.b_id, 5, "looks the same")
    assert c2 is None or (c2.a_id, c2.b_id) != (c1.a_id, c1.b_id)
    label = ctx.store.latest_human_label("sofas", c1.a_id, c1.b_id)
    assert label.cls == 5 and label.comment == "looks the same"
    if c2:
        assert c2.position == f"2/{total}"


def test_prepare_manual_test_and_import(app_ctx, tmp_path):
    from helpers import sofa_profile_with_root, write_small_csv
    from pseudolabel.profiles import dump_profile

    (app_ctx.settings.profiles_dir / "sofas.yaml").write_text(dump_profile(sofa_profile_with_root(tmp_path)), encoding="utf-8")
    msg = prepare_manual_test(app_ctx, "sofas", str(write_small_csv(tmp_path)), per_class=1)
    assert "3 pairs" in msg and len(app_ctx.store.queue_pairs("manual_test")) == 3
    h = tmp_path / "h.csv"
    h.write_text("a_id,b_id,class\n2272175750151,380288750116,5\n")
    assert "1" in import_csv(app_ctx, "sofas", str(h))
    assert import_csv(app_ctx, "sofas", str(tmp_path / "missing.csv")).startswith("Error")
