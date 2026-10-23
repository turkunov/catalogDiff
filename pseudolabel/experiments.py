"""Labeler ablations: run one variant over the eval set (per-pair JSONL + metrics) and compare variants."""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pandas as pd
import yaml

from pseudolabel.data import Universe, item_from_example, load_universe
from pseudolabel.labeler import Labeler, LabelerOptions, PairResult
from pseudolabel.metrics import auroc, bootstrap_diff, mcnemar_p, quadratic_kappa, require_threshold
from pseudolabel.profiles import Profile
from pseudolabel.store import Store

CONFIDENCE_LEVELS = (0.5, 0.7, 0.9)


@dataclass
class Variant:
    name: str
    base_url: str
    model: str
    profile_patch: dict = field(default_factory=dict)  # Profile fields to override, e.g. {"thinking": False}
    options: dict = field(default_factory=dict)  # LabelerOptions fields; few_shot: true uses the profile examples
    image_map: str | None = None  # JSON {original image path: replacement}, e.g. segmented photos
    skeptic_of: str | None = None  # skeptic pass over the results of that variant
    concurrency: int = 16


def load_config(path: Path) -> tuple[dict, dict[str, Variant]]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg, {d["name"]: Variant(**d) for d in cfg.pop("variants")}


def load_eval(pairs_csv: Path, labels_csv: Path, profile: Profile) -> tuple[Universe, dict[tuple[str, str], int]]:
    universe = load_universe(profile, Path(pairs_csv))
    df = pd.read_csv(labels_csv, dtype={"a_id": str, "b_id": str})
    return universe, {(r["a_id"], r["b_id"]): int(r["class"]) for r in df.to_dict("records")}


def _options(v: Variant, profile: Profile, namespace: str) -> LabelerOptions:
    o = dict(v.options)
    examples = []
    if o.pop("few_shot", False):
        root = Path(profile.image_root)
        examples = [(item_from_example("a", e.a, root), item_from_example("b", e.b, root), e.expected_class)
                    for e in profile.examples]
    return LabelerOptions(**o, few_shot=examples, namespace=namespace)


def read_rows(path: Path) -> dict[tuple[str, str], dict]:
    if not Path(path).exists():
        return {}
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(r["a_id"], r["b_id"]): r for r in rows}


def _first_result(row: dict) -> PairResult:
    probs = {int(k): v for k, v in row["probs"].items()} if row.get("probs") else None
    return PairResult(row["vlm"], row.get("reasoning", ""), "", row.get("error"), row.get("latency_s") or 0.0,
                      row.get("visual_a") or {}, row.get("visual_b") or {}, row.get("params_a") or {},
                      row.get("params_b") or {}, probs, row.get("confidence"), row.get("completion_tokens") or 0)


def variant_metrics(rows: list[dict], threshold: int) -> dict:
    t = require_threshold(threshold)
    human = [r["human"] for r in rows]
    pred = [r["vlm"] for r in rows]
    ok = [p == h for p, h in zip(pred, human)]
    answered = [(h, p) for h, p in zip(human, pred) if p is not None]
    pred_pos = [p is not None and p >= t for p in pred]
    true_pos = [h >= t for h in human]
    tp = sum(1 for x, y in zip(pred_pos, true_pos) if x and y)
    precision = tp / sum(pred_pos) if sum(pred_pos) else 0.0
    recall = tp / sum(true_pos) if sum(true_pos) else 0.0
    with_conf = [(r["confidence"], r["vlm"] == r["human"]) for r in rows if r["vlm"] is not None and r.get("confidence") is not None]
    n = len(rows)
    out = {
        "n": n,
        "failed": sum(p is None for p in pred),
        "exact": sum(ok) / n if n else 0.0,
        "within1": sum(1 for h, p in answered if abs(h - p) <= 1) / n if n else 0.0,
        "qwk": quadratic_kappa([h for h, _ in answered], [p for _, p in answered]) if answered else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "binary_accuracy": sum(1 for x, y in zip(pred_pos, true_pos) if x == y) / n if n else 0.0,
        "confusion": [[sum(1 for h, p in answered if h == i and p == j) for j in range(1, 6)] for i in range(1, 6)],
        "exact_by_class": {c: (sum(1 for h, o in zip(human, ok) if h == c and o) / human.count(c)) if human.count(c) else None
                           for c in range(1, 6)},
        "confidence_auroc": auroc([c for c, _ in with_conf], [o for _, o in with_conf]) if with_conf else None,
        "low_confidence_share": {str(lv): (sum(1 for c, _ in with_conf if c < lv) / len(with_conf)) if with_conf else None
                                 for lv in CONFIDENCE_LEVELS},
        "mean_latency_s": sum(r.get("latency_s") or 0.0 for r in rows) / n if n else 0.0,
        "match_tokens_per_pair": sum(r.get("completion_tokens") or 0 for r in rows) / n if n else 0.0,
    }
    return out


def run_variant(v: Variant, profile: Profile, universe: Universe, human: dict, store: Store, out_dir: Path,
                threshold: int, vlm=None, limit: int | None = None, progress=None, retry_failed: bool = False) -> dict:
    """Label the eval pairs not yet in out_dir/<name>.jsonl, then write <name>.metrics.json; returns the metrics.

    retry_failed also labels again the pairs whose last row has no class; the newer row wins.
    """
    require_threshold(threshold)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    namespace = f"{profile.name}@{v.name}"
    pv = replace(profile, name=namespace, **v.profile_patch)
    version = store.register_profile_version(pv)
    if vlm is None:
        from pseudolabel.vlm import OpenAIVLM

        vlm = OpenAIVLM(v.base_url, v.model)
    labeler = Labeler(vlm, store, pv, version, options=_options(v, profile, namespace))
    image_map = json.loads(Path(v.image_map).read_text()) if v.image_map else {}
    remap = (lambda it: replace(it, image=image_map.get(it.image, it.image))) if image_map else (lambda it: it)
    source = read_rows(out_dir / f"{v.skeptic_of}.jsonl") if v.skeptic_of else {}
    pairs = [p for p in universe.pairs if p in human and (not v.skeptic_of or p in source)][: limit or None]
    path = out_dir / f"{v.name}.jsonl"
    done = read_rows(path)
    todo = [p for p in pairs if p not in done or (retry_failed and done[p]["vlm"] is None)]
    lock, counter, start = threading.Lock(), [0], time.time()

    def one(pair: tuple[str, str]) -> None:
        a, b = remap(universe.items_a[pair[0]]), remap(universe.items_b[pair[1]])
        try:
            if v.skeptic_of:
                r = labeler.skeptic(a, b, _first_result(source[pair]), threshold)
            else:
                r = labeler.label_pair(a, b, use_cache=False)
        except Exception as e:  # e.g. the store: record the pair as failed and keep going
            r = PairResult(None, "", "", f"{type(e).__name__}: {e}", 0.0)
        row = {"a_id": pair[0], "b_id": pair[1], "human": human[pair], "vlm": r.cls, "probs": r.probs,
               "confidence": r.confidence, "latency_s": r.latency_s, "completion_tokens": r.completion_tokens,
               "error": r.error, "reasoning": r.reasoning, "visual_a": r.visual_a, "visual_b": r.visual_b,
               "params_a": r.params_a, "params_b": r.params_b, "skeptic_cls": r.skeptic_cls}
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counter[0] += 1
            if progress:
                progress(counter[0], len(todo))

    with ThreadPoolExecutor(max_workers=max(1, v.concurrency)) as pool:
        list(pool.map(one, todo))
    rows = [r for p, r in read_rows(path).items() if p in set(pairs)]
    metrics = {"variant": asdict(v), **variant_metrics(rows, threshold), "wall_s_last_run": time.time() - start,
               "pairs_labeled_last_run": len(todo), "tokens_all_calls_last_run": labeler.completion_tokens}
    (out_dir / f"{v.name}.metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _fmt(x, signed: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def report(out_dir: Path, baseline: str = "base", threshold: int = 4, order: list[str] | None = None) -> str:
    """Markdown tables: metrics per variant and paired comparison with the baseline on the same pairs."""
    t = require_threshold(threshold)
    out_dir = Path(out_dir)
    runs = {p.stem: read_rows(p) for p in sorted(out_dir.glob("*.jsonl"))}
    names = [n for n in (order or []) if n in runs] + sorted(n for n in runs if n not in (order or []))
    names = [baseline] + [n for n in names if n != baseline] if baseline in runs else names
    base = runs.get(baseline, {})
    lines = ["| variant | n | failed | exact | ±1 | QWK | P | R | F1 | conf AUROC | s/pair | Δexact vs base [95% CI] | McNemar p (binary) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    by_class = ["| variant | 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|"]
    for name in names:
        rows = runs[name]
        m = variant_metrics(list(rows.values()), t)
        if name == baseline:
            delta, p = "baseline", "—"
        else:
            common = [k for k in rows if k in base]
            b_ok = [base[k]["vlm"] == base[k]["human"] for k in common]
            v_ok = [rows[k]["vlm"] == rows[k]["human"] for k in common]
            pos = lambda r: r["vlm"] is not None and r["vlm"] >= t  # noqa: E731
            b_bin = [pos(base[k]) == (base[k]["human"] >= t) for k in common]
            v_bin = [pos(rows[k]) == (rows[k]["human"] >= t) for k in common]
            d = (sum(v_ok) - sum(b_ok)) / len(common) if common else None
            lo, hi = bootstrap_diff(b_ok, v_ok) if common else (None, None)
            delta = f"{_fmt(d, True)} [{_fmt(lo, True)}, {_fmt(hi, True)}]"
            p = f"{mcnemar_p(b_bin, v_bin):.3f}" if common else "—"
        lines.append(f"| {name} | {m['n']} | {m['failed']} | {_fmt(m['exact'])} | {_fmt(m['within1'])} | {_fmt(m['qwk'])} | "
                     f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {_fmt(m['confidence_auroc'])} | "
                     f"{m['mean_latency_s']:.1f} | {delta} | {p} |")
        by_class.append(f"| {name} | " + " | ".join(_fmt(m["exact_by_class"][c]) for c in range(1, 6)) + " |")
    return ("## Variants\n\n" + "\n".join(lines) + "\n\n## Exact match by reference class\n\n" + "\n".join(by_class) + "\n")


def photo_dup_rows(universe: Universe, human: dict, thresholds: list[int], base: dict) -> list[dict]:
    """Same-photo shortcut on the eval set: how many pairs it takes at each distance and how often they are 5 (or >= 4).

    With base rows, also the exact match of "base + shortcut" (shortcut pairs get 5, the rest keep the base class).
    """
    from pseudolabel.dedup import photo_distance

    dist = {p: photo_distance(universe.items_a[p[0]].image, universe.items_b[p[1]].image) for p in human}
    out = []
    for t in thresholds:
        hit = [p for p, d in dist.items() if d is not None and d <= t]
        row = {"max_distance": t, "pairs": len(hit),
               "precision_5": sum(human[p] == 5 for p in hit) / len(hit) if hit else None,
               "precision_4plus": sum(human[p] >= 4 for p in hit) / len(hit) if hit else None,
               "recall_5": sum(human[p] == 5 for p in hit) / max(1, sum(c == 5 for c in human.values()))}
        if base:
            pred = {p: (5 if p in set(hit) else r["vlm"]) for p, r in base.items()}
            row["base_plus_shortcut_exact"] = sum(pred[p] == human[p] for p in pred) / len(pred)
        out.append(row)
    return out


def stderr_progress(name: str):
    return lambda d, total: print(f"\r{name}: {d}/{total}", end="", file=sys.stderr, flush=True)
