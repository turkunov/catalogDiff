"""Fixed manual-test set: selection, human label import, and labeler-vs-human runs."""
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from pseudolabel.data import Universe
from pseudolabel.labeler import Labeler
from pseudolabel.metrics import labeler_vs_human
from pseudolabel.store import LabelRow, Store


def select_manual_test(universe: Universe, per_class: int = 40, seed: int = 0) -> tuple[list[tuple[str, str, int]], dict[int, int]]:
    """per_class pairs per legacy class 1..5, preferring Avito items not used yet; returns rows and shortfall."""
    rng = random.Random(seed)
    used: set[str] = set()
    rows: list[tuple[str, str, int]] = []
    shortfall: dict[int, int] = {}
    for cls in range(1, 6):
        pairs = [p for p, c in universe.legacy_labels.items() if c == cls]
        rng.shuffle(pairs)
        picked, seen = [], set()
        for a, b in pairs:
            if a not in used and a not in seen:
                picked.append((a, b))
                seen.add(a)
            if len(picked) == per_class:
                break
        taken = set(picked)
        picked += [p for p in pairs if p not in taken][: per_class - len(picked)]
        if len(picked) < per_class:
            shortfall[cls] = per_class - len(picked)
        used.update(a for a, _ in picked)
        rows += [(a, b, cls) for a, b in picked]
    return rows, shortfall


def load_manual_test_csv(path: Path) -> list[tuple[str, str, int]]:
    """A fixed set saved earlier (tools/manual_labeling output or a_id,b_id,legacy_class)."""
    df = pd.read_csv(path, lineterminator="\n", dtype=str)
    a_col = "a_id" if "a_id" in df.columns else "q_item_id"
    b_col = "b_id" if "b_id" in df.columns else "d_item_id"
    return [(r[a_col], r[b_col], int(r["legacy_class"])) for _, r in df.iterrows()]


def import_human_labels(store: Store, profile: str, csv_path: Path) -> int:
    """Import a_id,b_id,class[,comment]; extra columns are ignored; nothing is written if any row is invalid."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    rows = []
    for n, r in enumerate(df.to_dict("records"), start=1):
        try:
            cls = int(r["class"])
        except (KeyError, ValueError):
            cls = None
        if cls not in range(1, 6):
            raise ValueError(f"row {n}: class must be 1..5, got {r.get('class')!r}")
        rows.append(LabelRow(profile, r["a_id"], r["b_id"], "human", cls, comment=r.get("comment", "")))
    for row in rows:
        store.put_label(row)
    return len(rows)


def run_manual_test(store: Store, labeler: Labeler, universe: Universe, profile: str, threshold: int | None,
                    concurrency: int = 4) -> dict:
    pairs = store.manual_test_pairs(profile)

    def label(a_id: str, b_id: str):
        return labeler.label_pair(universe.items_a[a_id], universe.items_b[b_id])

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(lambda p: label(p[0], p[1]), pairs))
    rows = []
    for (a, b, _), res in zip(pairs, results):
        human = store.latest_human_label(profile, a, b)
        rows.append({"human": human.cls if human else None, "vlm": res.cls})
    metrics = labeler_vs_human(pd.DataFrame(rows, columns=["human", "vlm"]), threshold)
    metrics["mean_latency_s"] = float(sum(r.latency_s for r in results) / len(results)) if results else 0.0
    metrics["pairs"] = len(pairs)
    return metrics
