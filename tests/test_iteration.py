import pytest

from pseudolabel.iteration import finalize_iteration, plan_iteration, run_iteration


def selected(store, it):
    return store.iteration_pairs(it).query("selected_for_vlm == 1")


def test_plan_respects_budget_and_strata(env):
    it = plan_iteration(env.cfg(budget=10), env.store, env.profile, env.universe, env.encoder, env.tmp)
    pairs = env.store.iteration_pairs(it)
    assert (pairs.selected_for_vlm == 1).sum() == 10
    assert set(pairs.origin) <= {"top_1_10", "top_11_100", "top_101_1000", "random"}
    assert env.store.get_iteration(it)["encoder_version"] == "fake@1"
    assert env.store.get_item("sofas", "a", "a0") is not None


def test_plan_easy_negatives_partly_checked(env):
    it = plan_iteration(env.cfg(budget=4, easy_negatives=10, easy_negative_check_share=0.2),
                        env.store, env.profile, env.universe, env.encoder, env.tmp)
    negs = env.store.iteration_pairs(it).query("origin == 'easy_negative'")
    assert len(negs) == 10 and negs.selected_for_vlm.sum() == 2
    assert set(negs.query("selected_for_vlm == 0").status) == {"not_selected"}


def test_second_iteration_skips_labeled(env):
    it1 = plan_iteration(env.cfg(budget=5), env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it1, env.store, env.labeler)
    it2 = plan_iteration(env.cfg(budget=5), env.store, env.profile, env.universe, env.encoder, env.tmp)
    done1 = set(map(tuple, env.store.iteration_pairs(it1)[["a_id", "b_id"]].values))
    sel2 = selected(env.store, it2)
    assert len(sel2) == 5 and not done1 & set(map(tuple, sel2[["a_id", "b_id"]].values))


def test_resume_does_not_relabel(env):
    it = plan_iteration(env.cfg(budget=6), env.store, env.profile, env.universe, env.encoder, env.tmp)
    first = env.store.iteration_pairs(it).iloc[0]
    env.labeler.label_pair(env.universe.items_a[first.a_id], env.universe.items_b[first.b_id])
    env.store.set_iteration_pair_status(it, first.a_id, first.b_id, "done")
    calls_before = env.match_calls()
    run_iteration(it, env.store, env.labeler)
    assert env.match_calls() - calls_before == 5
    assert set(env.store.iteration_pairs(it).status) == {"done"}


def test_run_rejects_labeler_with_other_profile_version(env):
    it = plan_iteration(env.cfg(budget=2), env.store, env.profile, env.universe, env.encoder, env.tmp)
    env.labeler.profile_version = 99
    with pytest.raises(ValueError, match="profile version"):
        run_iteration(it, env.store, env.labeler)


def test_iteration_continues_after_pair_error(env_with_one_broken_image):
    env = env_with_one_broken_image
    it = plan_iteration(env.cfg(budget=8), env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it, env.store, env.labeler)
    st = selected(env.store, it).status
    assert (st == "error").sum() >= 1 and (st == "done").sum() >= 1 and (st == "pending").sum() == 0


def test_finalize_report_and_disagreement_queue(env):
    it = plan_iteration(env.cfg(budget=10, positive_threshold=4), env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it, env.store, env.labeler)  # the fake answers class 2 for every pair
    rep = finalize_iteration(it, env.store)
    assert rep["threshold_missing"] is False
    assert rep["origin_stats"]["top_1_10"]["disagreements"] == rep["origin_stats"]["top_1_10"]["n"] > 0
    assert env.store.queue_pairs(f"disagreements:{it}")
    assert env.store.get_iteration(it)["report"]["origin_stats"] == rep["origin_stats"]


def test_finalize_without_threshold(env):
    it = plan_iteration(env.cfg(budget=4), env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it, env.store, env.labeler)
    rep = finalize_iteration(it, env.store)
    assert rep["threshold_missing"] is True and "disagreements" not in rep["origin_stats"]["top_1_10"]
    assert env.store.queues("sofas") == []


def test_finalize_delta_and_manual_test(env):
    from pseudolabel.store import LabelRow

    it1 = plan_iteration(env.cfg(budget=4, positive_threshold=4), env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it1, env.store, env.labeler)
    finalize_iteration(it1, env.store)
    p = selected(env.store, it1).iloc[0]
    env.store.set_manual_test("sofas", [(p.a_id, p.b_id, 5)])
    env.store.put_label(LabelRow("sofas", p.a_id, p.b_id, "human", 2))
    it2 = plan_iteration(env.cfg(budget=4, positive_threshold=4), env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it2, env.store, env.labeler)
    rep = finalize_iteration(it2, env.store)
    assert "top_1_10" in rep["delta"]
    assert rep["manual_test"]["n"] == 1 and rep["manual_test"]["exact"] == 1.0


def test_head_not_supported(env):
    with pytest.raises(NotImplementedError, match="head"):
        plan_iteration(env.cfg(head="some/head"), env.store, env.profile, env.universe, env.encoder, env.tmp)


def test_stop_max_iterations(env):
    for _ in range(2):
        it = plan_iteration(env.cfg(budget=2, stop={"kind": "max_iterations", "value": 2}), env.store, env.profile,
                            env.universe, env.encoder, env.tmp)
        run_iteration(it, env.store, env.labeler)
        rep = finalize_iteration(it, env.store)
    assert rep["stop"]["triggered"] is True


def test_stop_min_new_signal(env):
    it = plan_iteration(env.cfg(budget=4, positive_threshold=4, stop={"kind": "min_new_signal", "value": 1}),
                        env.store, env.profile, env.universe, env.encoder, env.tmp)
    run_iteration(it, env.store, env.labeler)
    rep = finalize_iteration(it, env.store)
    signal = sum(s.get("positives", 0) + s.get("disagreements", 0) for s in rep["origin_stats"].values())
    assert rep["stop"]["triggered"] is (signal < 1)


def test_encoder_search_mode_has_title_origin(env, monkeypatch):
    import pseudolabel.sampling as sampling

    monkeypatch.setattr(sampling, "TOP_K", 5)  # emulate a catalog much larger than the retrieval depth
    it = plan_iteration(env.cfg(budget=20, mode="encoder_search"), env.store, env.profile, env.universe, env.encoder, env.tmp)
    assert "title" in set(selected(env.store, it).origin)


def test_run_rejects_labeler_with_edited_profile_same_version(env):
    it = plan_iteration(env.cfg(budget=2), env.store, env.profile, env.universe, env.encoder, env.tmp)
    env.labeler.profile.match_rules += " edited after planning"
    with pytest.raises(ValueError, match="profile"):
        run_iteration(it, env.store, env.labeler)


def test_run_rejects_labeler_with_other_model(env):
    it = plan_iteration(env.cfg(budget=2), env.store, env.profile, env.universe, env.encoder, env.tmp)
    env.vlm.model = "another-model"
    with pytest.raises(ValueError, match="model"):
        run_iteration(it, env.store, env.labeler)
