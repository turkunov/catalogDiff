"""Exact cosine retrieval over L2-normalized embeddings, rank positions and strata."""
import numpy as np

STRATA = (("top_1_10", 1, 10), ("top_11_100", 11, 100), ("top_101_1000", 101, 1000))


def stratum_for_rank(rank: int) -> str | None:
    for name, lo, hi in STRATA:
        if lo <= rank <= hi:
            return name
    return None


def pair_positions(emb_a: np.ndarray, emb_b: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1-based rank of b among all B for its a (1 + number of B with a strictly higher cosine), and the cosine."""
    a_idx, b_idx = np.asarray(a_idx), np.asarray(b_idx)
    ranks = np.empty(len(a_idx), dtype=np.int64)
    cos = np.empty(len(a_idx), dtype=np.float32)
    for a in np.unique(a_idx):
        rows = np.flatnonzero(a_idx == a)
        sims = emb_b @ emb_a[a]
        target = sims[b_idx[rows]]
        cos[rows] = target
        ranks[rows] = 1 + (sims[None, :] > target[:, None]).sum(axis=1)
    return ranks, cos


def top_k(emb_a: np.ndarray, emb_b: np.ndarray, k: int = 1000, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosines of the k nearest B for every A, sorted by decreasing cosine."""
    k = min(k, len(emb_b))
    idx_out = np.empty((len(emb_a), k), dtype=np.int64)
    cos_out = np.empty((len(emb_a), k), dtype=np.float32)
    for start in range(0, len(emb_a), chunk):
        sims = emb_a[start:start + chunk] @ emb_b.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k < sims.shape[1] else np.tile(np.arange(k), (len(sims), 1))
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1, kind="stable")
        idx_out[start:start + chunk] = np.take_along_axis(part, order, axis=1)
        cos_out[start:start + chunk] = np.take_along_axis(part_sims, order, axis=1)
    return idx_out, cos_out
