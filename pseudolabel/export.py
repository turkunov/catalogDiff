"""Training-set export: binary labels by an explicit threshold, human labels over VLM, auto negatives."""
from pathlib import Path

import pandas as pd

from pseudolabel.metrics import ThresholdRequired, require_threshold
from pseudolabel.store import Store

__all__ = ["ThresholdRequired", "build_training_set", "write_export"]

COLUMNS = ["a_id", "b_id", "a_image", "b_image", "raw_class", "label", "weight", "label_source", "origin",
           "iteration_id", "encoder_version", "profile_version"]


def _latest(labels: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {(r.a_id, r.b_id): int(r.cls) for r in labels.drop_duplicates(["a_id", "b_id"], keep="last").itertuples()}


def build_training_set(store: Store, profile: str, threshold: int | None, class_weights: dict[int, float] | None = None,
                       iteration_ids: list[int] | None = None, include_legacy: bool = False) -> pd.DataFrame:
    t = require_threshold(threshold)
    weights = class_weights or {}
    labels = store.query_labels(profile)
    labels = labels[labels["cls"].notna() & labels["error"].isna()]
    human = _latest(labels[labels["source"] == "human"])
    iterations = store.list_iterations(profile)
    if iteration_ids is not None:
        iterations = iterations[iterations["id"].isin(iteration_ids)]

    rows: dict[tuple[str, str], dict] = {}
    for it in iterations.itertuples():
        config = store.get_iteration(it.id)["config"]
        vlm = labels[labels["source"].isin(["vlm", "photo_dup"]) & (labels["profile_version"] == it.profile_version)]
        if config.get("labeler_model"):
            vlm = vlm[vlm["model"] == config["labeler_model"]]
        vlm_source = {(r.a_id, r.b_id): r.source for r in vlm.drop_duplicates(["a_id", "b_id"], keep="last").itertuples()}
        vlm = _latest(vlm)
        for p in store.iteration_pairs(it.id).itertuples():
            key = (p.a_id, p.b_id)
            base = {"origin": p.origin, "iteration_id": it.id, "encoder_version": it.encoder_version,
                    "profile_version": it.profile_version}
            if key in human:
                rows[key] = {**base, "raw_class": human[key], "label_source": "human"}
            elif p.origin == "easy_negative":
                rows[key] = {**base, "raw_class": None, "label_source": "auto_negative"}
            elif key in vlm:
                rows[key] = {**base, "raw_class": vlm[key], "label_source": vlm_source[key]}
    if include_legacy:
        for key, cls in _latest(labels[labels["source"] == "legacy"]).items():
            rows.setdefault(key, {"origin": "legacy", "iteration_id": None, "encoder_version": None,
                                  "profile_version": None, "raw_class": human.get(key, cls),
                                  "label_source": "human" if key in human else "legacy"})

    out = []
    for (a, b), r in rows.items():
        cls = r["raw_class"]
        item_a, item_b = store.get_item(profile, "a", a), store.get_item(profile, "b", b)
        out.append({"a_id": a, "b_id": b, "a_image": item_a.image if item_a else None,
                    "b_image": item_b.image if item_b else None, **r,
                    "label": int(cls is not None and cls >= t),
                    "weight": 1.0 if cls is None else float(weights.get(cls, 1.0))})
    df = pd.DataFrame(out, columns=COLUMNS)
    return df.astype({"raw_class": "Int64"})


def write_export(df: pd.DataFrame, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq, csv = out_dir / f"{stem}.parquet", out_dir / f"{stem}.csv"
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)
    return pq, csv
