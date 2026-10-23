"""Labeler-vs-human metrics, the disagreement rule and per-origin statistics."""
import pandas as pd

THRESHOLD_MISSING = "Positive class threshold is not set"
DISAGREEMENT_HIGH = {"top_1_10"}
DISAGREEMENT_LOW = {"top_11_100", "top_101_1000", "title"}


class ThresholdRequired(ValueError):
    def __init__(self):
        super().__init__(THRESHOLD_MISSING)


def require_threshold(threshold: int | None) -> int:
    if threshold is None:
        raise ThresholdRequired()
    if int(threshold) not in range(2, 6):
        raise ValueError(f"positive class threshold must be 2..5, got {threshold}")
    return int(threshold)


def labeler_vs_human(df: pd.DataFrame, threshold: int | None = None) -> dict:
    both = df.dropna(subset=["human", "vlm"]).astype({"human": int, "vlm": int})
    diff = (both["human"] - both["vlm"]).abs()
    confusion = [[int(((both["human"] == h) & (both["vlm"] == v)).sum()) for v in range(1, 6)] for h in range(1, 6)]
    out = {
        "n": int(len(both)),
        "exact": float((diff == 0).mean()) if len(both) else 0.0,
        "within1": float((diff <= 1).mean()) if len(both) else 0.0,
        "confusion": confusion,
        "vlm_failed": int(df["vlm"].isna().sum()),
    }
    if threshold is not None:
        t = require_threshold(threshold)
        pred, true = both["vlm"] >= t, both["human"] >= t
        tp = int((pred & true).sum())
        out["precision"] = tp / int(pred.sum()) if pred.sum() else 0.0
        out["recall"] = tp / int(true.sum()) if true.sum() else 0.0
    return out


def is_disagreement(origin: str, cls: int | None, threshold: int) -> bool:
    if cls is None or pd.isna(cls):
        return False
    if origin in DISAGREEMENT_HIGH:
        return int(cls) < threshold
    if origin in DISAGREEMENT_LOW:
        return int(cls) >= threshold
    return False


def origin_stats(df: pd.DataFrame, threshold: int | None) -> dict[str, dict]:
    stats = {}
    for origin, part in df.groupby("origin"):
        cls = part["cls"]
        s = {
            "n": int(len(part)),
            "class_counts": {str(k): int((cls == k).sum()) for k in range(1, 6)},  # str keys survive JSON
            "errors": int(cls.isna().sum()),
        }
        if threshold is not None:
            s["positives"] = int((cls >= threshold).sum())
            s["disagreements"] = int(sum(is_disagreement(origin, c, threshold) for c in cls))
        stats[str(origin)] = s
    return stats


def quadratic_kappa(y_true: list[int], y_pred: list[int], k: int = 5) -> float:
    """Cohen's kappa with quadratic weights for classes 1..k: errors far apart on the scale cost more."""
    import numpy as np

    o = np.zeros((k, k))
    for t, p in zip(y_true, y_pred):
        o[t - 1, p - 1] += 1
    if o.sum() == 0:
        return 0.0
    idx = np.arange(k)
    w = (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    denom = (w * e).sum()
    return 1.0 if denom == 0 else float(1 - (w * o).sum() / denom)


def mcnemar_p(base_ok: list[bool], var_ok: list[bool]) -> float:
    """Exact two-sided McNemar test on paired correctness."""
    from math import comb

    b = sum(1 for x, y in zip(base_ok, var_ok) if x and not y)
    c = sum(1 for x, y in zip(base_ok, var_ok) if y and not x)
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def bootstrap_diff(base_ok: list[bool], var_ok: list[bool], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of mean(var_ok) - mean(base_ok) over the same pairs."""
    import numpy as np

    d = np.asarray(var_ok, dtype=float) - np.asarray(base_ok, dtype=float)
    if len(d) == 0:
        return 0.0, 0.0
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    means = d[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def auroc(scores: list[float], positive: list[bool]) -> float | None:
    """Probability that a random positive scores higher than a random negative (ties count half)."""
    pos = [s for s, y in zip(scores, positive) if y]
    neg = [s for s, y in zip(scores, positive) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))
