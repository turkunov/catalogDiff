"""Candidate pools by retrieval strata, budget allocation, the title source and easy negatives.

Embedding matrices are aligned with `list(universe.items_a)` / `list(universe.items_b)`.
"""
import math
import random
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pseudolabel.data import Item, Universe
from pseudolabel.retrieval import pair_positions, stratum_for_rank, top_k

DEFAULT_SHARES = {"top_1_10": 0.60, "top_11_100": 0.25, "top_101_1000": 0.10, "random": 0.05}
DEFAULT_TITLE_SHARE = 0.10
POOL_ORDER = ("top_1_10", "top_11_100", "top_101_1000", "title", "random")
TOP_K = 1000
RANDOM_PER_QUERY = 20


@dataclass
class Candidate:
    a_id: str
    b_id: str
    origin: str
    rank: int | None
    cosine: float | None


def allocate_budget(budget: int, shares: dict[str, float], available: dict[str, int]) -> dict[str, int]:
    """Split `budget` by shares (largest remainder), capped by availability; shortfalls go to pools with room."""
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise ValueError(f"shares must sum to 1, got {sum(shares.values()):.6f}")
    alloc = {k: 0 for k in shares}
    total = min(budget, sum(available.get(k, 0) for k, s in shares.items() if s > 0))
    active = [k for k, s in shares.items() if s > 0 and available.get(k, 0) > 0]
    while sum(alloc.values()) < total and active:
        remaining = total - sum(alloc.values())
        weight = sum(shares[k] for k in active)
        quotas = {k: remaining * shares[k] / weight for k in active}
        give = {k: math.floor(q) for k, q in quotas.items()}
        by_fraction = sorted(active, key=lambda k: (quotas[k] - give[k], shares[k]), reverse=True)
        for k in by_fraction[: remaining - sum(give.values())]:
            give[k] += 1
        for k in active:
            alloc[k] += min(give[k], available[k] - alloc[k])
        active = [k for k in active if alloc[k] < available[k]]
    return alloc


def effective_shares(mode: str, shares: dict[str, float], title_share: float | None) -> dict[str, float]:
    if mode != "encoder_search":
        return dict(shares)
    t = DEFAULT_TITLE_SHARE if title_share is None else title_share
    return {**{k: v * (1 - t) for k, v in shares.items()}, "title": t}


def title_candidates(items_a: dict[str, Item], items_b: dict[str, Item], top_n: int = 10) -> list[tuple[str, str, float]]:
    """Encoder-independent source: nearest titles by TF-IDF over character n-grams."""
    if not items_a or not items_b:
        return []
    ids_a, ids_b = list(items_a), list(items_b)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    vec.fit([i.title for i in items_a.values()] + [i.title for i in items_b.values()])
    sims = (vec.transform([items_a[i].title for i in ids_a]) @ vec.transform([items_b[i].title for i in ids_b]).T).toarray()
    out = []
    for r, a_id in enumerate(ids_a):
        for c in np.argsort(-sims[r])[:top_n]:
            if sims[r, c] > 0:
                out.append((a_id, ids_b[c], float(sims[r, c])))
    return out


def _positions(universe: Universe, emb_a, emb_b, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    a_index = {k: i for i, k in enumerate(universe.items_a)}
    b_index = {k: i for i, k in enumerate(universe.items_b)}
    if not pairs:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    return pair_positions(emb_a, emb_b, np.array([a_index[a] for a, _ in pairs]), np.array([b_index[b] for _, b in pairs]))


def build_pools(universe: Universe, emb_a: np.ndarray, emb_b: np.ndarray, mode: str,
                exclude_pairs: set[tuple[str, str]], exclude_a: set[str], title_top_n: int = 10,
                seed: int = 0) -> dict[str, list[Candidate]]:
    """Candidate pools; every pair lands in exactly one pool, the first in POOL_ORDER that has it."""
    raw: dict[str, list[tuple[str, str]]] = {k: [] for k in POOL_ORDER}
    ids_a, ids_b = list(universe.items_a), list(universe.items_b)
    if mode == "pairs_file":
        pairs = [p for p in universe.pairs if p[0] not in exclude_a]
        ranks, _ = _positions(universe, emb_a, emb_b, pairs)
        for p, r in zip(pairs, ranks):
            raw[stratum_for_rank(int(r)) or "random"].append(p)
    elif mode == "encoder_search":
        rng = random.Random(seed)
        queries = [i for i, a in enumerate(ids_a) if a not in exclude_a]
        idx, _ = top_k(emb_a[queries], emb_b, k=TOP_K)
        for row, qi in enumerate(queries):
            top = idx[row].tolist()
            for pos, bi in enumerate(top, start=1):
                raw[stratum_for_rank(pos)].append((ids_a[qi], ids_b[bi]))
            outside = sorted(set(range(len(ids_b))) - set(top))
            for bi in rng.sample(outside, min(RANDOM_PER_QUERY, len(outside))):
                raw["random"].append((ids_a[qi], ids_b[bi]))
        kept_a = {a: universe.items_a[a] for a in ids_a if a not in exclude_a}
        raw["title"] = [(a, b) for a, b, _ in title_candidates(kept_a, universe.items_b, title_top_n)]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    seen = set(exclude_pairs)
    pools: dict[str, list[Candidate]] = {}
    for origin in POOL_ORDER:
        pairs = [p for p in dict.fromkeys(raw[origin]) if p not in seen]
        seen.update(pairs)
        ranks, cos = _positions(universe, emb_a, emb_b, pairs)
        pools[origin] = [Candidate(a, b, origin, int(r), float(c)) for (a, b), r, c in zip(pairs, ranks, cos)]
    return pools


def select_pairs(pools: dict[str, list[Candidate]], budget: int, shares: dict[str, float], seed: int) -> list[Candidate]:
    rng = random.Random(seed)
    alloc = allocate_budget(budget, shares, {k: len(pools.get(k, [])) for k in shares})
    chosen: list[Candidate] = []
    for origin, n in alloc.items():
        chosen += rng.sample(pools.get(origin, []), n)
    return chosen


def sample_easy_negatives(universe: Universe, emb_a: np.ndarray, emb_b: np.ndarray, n: int,
                          exclude_pairs: set, seed: int, exclude_a: set[str] | None = None) -> list[Candidate]:
    """Random pairs ranked beyond the top 1000 (beyond the top half for catalogs smaller than 2000)."""
    rng = random.Random(seed)
    ids_a = [a for a in universe.items_a if a not in (exclude_a or set())]
    ids_b = list(universe.items_b)
    if not ids_a or not ids_b:
        return []
    threshold = min(TOP_K, len(ids_b) // 2)
    out: list[Candidate] = []
    taken = set(exclude_pairs)
    for _ in range(n * 50):
        if len(out) >= n:
            break
        pair = (rng.choice(ids_a), rng.choice(ids_b))
        if pair in taken:
            continue
        taken.add(pair)
        ranks, cos = _positions(universe, emb_a, emb_b, [pair])
        if ranks[0] > threshold:
            out.append(Candidate(pair[0], pair[1], "easy_negative", int(ranks[0]), float(cos[0])))
    return out
