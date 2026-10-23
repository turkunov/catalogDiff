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
            "class_counts": {k: int((cls == k).sum()) for k in range(1, 6)},
            "errors": int(cls.isna().sum()),
        }
        if threshold is not None:
            s["positives"] = int((cls >= threshold).sum())
            s["disagreements"] = int(sum(is_disagreement(origin, c, threshold) for c in cls))
        stats[str(origin)] = s
    return stats
