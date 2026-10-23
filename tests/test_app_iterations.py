import pandas as pd

from pseudolabel.app.tab_iterations import export_iteration, iterations_table, parse_weights, start_iteration, wait


def test_export_iteration_without_threshold(app_ctx_with_iteration):
    paths, msg = export_iteration(app_ctx_with_iteration, "sofas", [app_ctx_with_iteration.iteration_id], None, "")
    assert paths is None and msg == "Positive class threshold is not set"


def test_export_iteration_writes_files(app_ctx_with_iteration):
    path, msg = export_iteration(app_ctx_with_iteration, "sofas", [app_ctx_with_iteration.iteration_id], 4, "3:0.5")
    assert path and path.endswith(".parquet") and len(pd.read_parquet(path)) > 0


def test_parse_weights():
    assert parse_weights("3:0.5, 4:0.8") == {3: 0.5, 4: 0.8}
    assert parse_weights("") == {}
    try:
        parse_weights("three=half")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "3:0.5" in str(e)


def test_iterations_table_counts(app_ctx_with_iteration):
    t = iterations_table(app_ctx_with_iteration, "sofas")
    assert {"done", "pending", "error", "positives", "disagreements"} <= set(t.columns)
    assert int(t.iloc[0]["done"]) == 8 and int(t.iloc[0]["pending"]) == 0


def test_start_iteration_runs_in_background(app_ctx, tmp_path):
    from fakes import FakeEncoder, route_by_prompt
    from helpers import make_jpeg, sofa_profile_with_root, write_small_csv
    from pseudolabel.data import load_universe
    from pseudolabel.profiles import dump_profile

    profile = sofa_profile_with_root(tmp_path)
    (app_ctx.settings.profiles_dir / "sofas.yaml").write_text(dump_profile(profile), encoding="utf-8")
    pairs = write_small_csv(tmp_path)
    universe = load_universe(profile, pairs)
    for i, it in enumerate(list(universe.items_a.values()) + list(universe.items_b.values())):
        make_jpeg(tmp_path, color=(i * 30, 5, 5), name=str(__import__("pathlib").Path(it.image).relative_to(tmp_path)))
    app_ctx.encoder_factory = lambda ckpt: FakeEncoder(
        {it.image: [1, i] for i, it in enumerate(list(universe.items_a.values()) + list(universe.items_b.values()))})
    app_ctx.vlm.handler = route_by_prompt(match='{"reasoning": "r", "class": 4}')
    it = start_iteration(app_ctx, {"profile": "sofas", "pairs_path": str(pairs), "mode": "pairs_file", "budget": 3,
                                   "positive_threshold": 4, "labeler_model": "fake-vlm"})
    wait(it, timeout=30)
    rep = app_ctx.store.get_iteration(it)["report"]
    assert rep is not None and rep["threshold_missing"] is False
    assert set(app_ctx.store.iteration_pairs(it).status) == {"done"}


def test_resume_uses_planned_profile_version(app_ctx_with_iteration):
    from fakes import prompt_text
    from pseudolabel.app.tab_iterations import resume_iteration, wait
    from pseudolabel.app.tab_profiles import load_profile_text, save_profile

    ctx = app_ctx_with_iteration
    it = ctx.iteration_id
    pending = ctx.store.iteration_pairs(it).iloc[0]
    ctx.store.set_iteration_pair_status(it, pending.a_id, pending.b_id, "error")
    # drop the stored label so the resumed pair needs a fresh match call
    ctx.store._write("DELETE FROM labels WHERE a_id=? AND b_id=?", (pending.a_id, pending.b_id))
    text = load_profile_text(ctx, "sofas").replace("category_label: Диваны", "category_label: EDITED", 1)
    assert save_profile(ctx, "sofas", text)[0]
    before = len(ctx.vlm.calls)
    resume_iteration(ctx, it)
    wait(it, timeout=30)
    new_prompts = [prompt_text(c["content"]) for c in ctx.vlm.calls[before:]]
    assert new_prompts and not any("EDITED" in p for p in new_prompts)


def test_background_failure_is_recorded(app_ctx_with_iteration, monkeypatch):
    import pseudolabel.app.tab_iterations as ti

    ctx = app_ctx_with_iteration
    def boom(*a, **k):
        raise RuntimeError("vLLM is down")
    monkeypatch.setattr(ti, "run_iteration", boom)
    ti.resume_iteration(ctx, ctx.iteration_id)
    ti.wait(ctx.iteration_id, timeout=10)
    it = ctx.store.get_iteration(ctx.iteration_id)
    assert it["status"] == "failed" and "vLLM is down" in it["report"]["error"]


def test_running_without_thread_shows_interrupted(app_ctx_with_iteration):
    ctx = app_ctx_with_iteration
    ctx.store.set_iteration_status(ctx.iteration_id, "running")
    t = iterations_table(ctx, "sofas")
    assert t.iloc[0]["status"] == "interrupted"


def test_iterations_table_eta_column(app_ctx_with_iteration):
    t = iterations_table(app_ctx_with_iteration, "sofas")
    assert "eta_min" in t.columns and t.iloc[0]["eta_min"] is None
