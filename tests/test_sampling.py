import pytest

from pseudolabel.data import Item
from pseudolabel.sampling import (
    DEFAULT_SHARES,
    allocate_budget,
    build_pools,
    effective_shares,
    sample_easy_negatives,
    select_pairs,
    title_candidates,
)


def item(side, id_, title):
    return Item(id_, side, title, "", {}, "")


def test_allocate_default():
    assert allocate_budget(100, DEFAULT_SHARES, {k: 1000 for k in DEFAULT_SHARES}) == {
        "top_1_10": 60, "top_11_100": 25, "top_101_1000": 10, "random": 5}


def test_allocate_redistributes_shortfall():
    out = allocate_budget(100, DEFAULT_SHARES, {"top_1_10": 1000, "top_11_100": 1000, "top_101_1000": 2, "random": 1000})
    assert out["top_101_1000"] == 2 and sum(out.values()) == 100
    assert out["top_1_10"] >= 60 and out["top_11_100"] >= 25 and out["random"] >= 5


def test_allocate_caps_at_available():
    out = allocate_budget(100, DEFAULT_SHARES, {"top_1_10": 3, "top_11_100": 0, "top_101_1000": 0, "random": 1})
    assert out == {"top_1_10": 3, "top_11_100": 0, "top_101_1000": 0, "random": 1}


def test_allocate_bad_shares():
    with pytest.raises(ValueError):
        allocate_budget(10, {"top_1_10": 0.5}, {"top_1_10": 10})


def test_effective_shares_title():
    s = effective_shares("encoder_search", DEFAULT_SHARES, None)
    assert s["title"] == pytest.approx(0.10) and s["top_1_10"] == pytest.approx(0.54)
    assert sum(s.values()) == pytest.approx(1.0)
    assert effective_shares("pairs_file", DEFAULT_SHARES, None) == DEFAULT_SHARES


def test_build_pools_excludes(tiny_universe, tiny_embeddings):
    pools = build_pools(tiny_universe, *tiny_embeddings, mode="pairs_file",
                        exclude_pairs={tiny_universe.pairs[0]}, exclude_a={"a3"})
    all_pairs = [(c.a_id, c.b_id) for pool in pools.values() for c in pool]
    assert tiny_universe.pairs[0] not in all_pairs
    assert not any(a == "a3" for a, _ in all_pairs)
    assert len(all_pairs) == len(set(all_pairs)) == 6 * 3 - 1


def test_build_pools_pairs_file_ranks_among_all_b(tiny_universe, tiny_embeddings):
    pools = build_pools(tiny_universe, *tiny_embeddings, mode="pairs_file", exclude_pairs=set(), exclude_a=set())
    ranks = [c.rank for pool in pools.values() for c in pool]
    assert max(ranks) > 10 and min(ranks) >= 1  # 30 B per query, so some file pairs land beyond the top 10
    assert all(c.rank <= 10 for c in pools["top_1_10"]) and pools.get("random", []) == []


def test_build_pools_encoder_search_unique_and_title(tiny_universe, tiny_embeddings):
    pools = build_pools(tiny_universe, *tiny_embeddings, mode="encoder_search", exclude_pairs=set(), exclude_a=set(), title_top_n=2)
    all_pairs = [(c.a_id, c.b_id) for pool in pools.values() for c in pool]
    assert len(all_pairs) == len(set(all_pairs))
    assert len(pools["top_1_10"]) == 4 * 10 and len(pools["top_11_100"]) == 4 * 20


def test_select_pairs_respects_allocation(tiny_universe, tiny_embeddings):
    pools = build_pools(tiny_universe, *tiny_embeddings, mode="encoder_search", exclude_pairs=set(), exclude_a=set())
    chosen = select_pairs(pools, 20, effective_shares("encoder_search", DEFAULT_SHARES, 0.0), seed=1)
    assert len(chosen) == 20 and len({(c.a_id, c.b_id) for c in chosen}) == 20


def test_title_candidates_finds_same_model():
    a = {"1": item("a", "1", "Диван Роденго серый")}
    b = {"x": item("b", "x", "диван кухонный роденго"), "y": item("b", "y", "пуф куб красный")}
    assert title_candidates(a, b, top_n=1)[0][:2] == ("1", "x")


def test_easy_negatives_are_low_ranked(tiny_universe, tiny_embeddings):
    negs = sample_easy_negatives(tiny_universe, *tiny_embeddings, n=3, exclude_pairs=set(), seed=0)
    assert len(negs) == 3 and all(c.origin == "easy_negative" for c in negs)
    assert all(c.rank > 15 for c in negs)  # bottom half of 30 candidates
