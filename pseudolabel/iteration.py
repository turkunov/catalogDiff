"""One loop iteration: plan (embeddings → pools → selection), resumable labeling, report and stop criteria."""
import math
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from pseudolabel.data import Universe
from pseudolabel.encoder import ImageEncoder, embed_items
from pseudolabel.labeler import Labeler
from pseudolabel.metrics import is_disagreement, labeler_vs_human, origin_stats, require_threshold
from pseudolabel.profiles import Profile, profile_hash
from pseudolabel.sampling import DEFAULT_SHARES, build_pools, effective_shares, sample_easy_negatives, select_pairs
from pseudolabel.store import Store


@dataclass
class IterationConfig:
    profile_name: str
    pairs_path: str
    mode: str
    encoder_checkpoint: str = "google/siglip2-so400m-patch14-384"
    head: str | None = None
    labeler_model: str | None = None
    budget: int = 200
    shares: dict = field(default_factory=lambda: dict(DEFAULT_SHARES))
    title_share: float | None = None
    positive_threshold: int | None = None
    easy_negatives: int = 0
    easy_negative_check_share: float = 0.02
    exclude_a_ids: list[str] = field(default_factory=list)
    review_size: int = 50
    stop: dict | None = None
    seed: int = 0


def _labeled_pairs(store: Store, profile: str, model: str, version: int) -> set[tuple[str, str]]:
    df = store.query_labels(profile, source="vlm", model=model, profile_version=version)
    df = df[df["error"].isna() & df["cls"].notna()]
    return set(zip(df["a_id"], df["b_id"]))


def plan_iteration(cfg: IterationConfig, store: Store, profile: Profile, universe: Universe,
                   encoder: ImageEncoder, cache_dir: Path) -> int:
    if cfg.head:
        raise NotImplementedError("Encoder head is not supported in v1")
    if not cfg.labeler_model:
        raise ValueError("labeler_model must be resolved before planning an iteration")
    if cfg.positive_threshold is not None:
        require_threshold(cfg.positive_threshold)
    version = store.register_profile_version(profile)
    store.upsert_items(profile.name, list(universe.items_a.values()) + list(universe.items_b.values()))
    store.import_legacy(profile.name, universe.legacy_labels)

    emb_a = embed_items(encoder, list(universe.items_a.values()), cache_dir)
    emb_b = embed_items(encoder, list(universe.items_b.values()), cache_dir)
    labeled = _labeled_pairs(store, profile.name, cfg.labeler_model, version)
    pools = build_pools(universe, emb_a, emb_b, cfg.mode, exclude_pairs=labeled, exclude_a=set(cfg.exclude_a_ids),
                        seed=cfg.seed)
    chosen = select_pairs(pools, cfg.budget, effective_shares(cfg.mode, cfg.shares, cfg.title_share), cfg.seed)
    rows = [dict(a_id=c.a_id, b_id=c.b_id, origin=c.origin, rank=c.rank, cosine=c.cosine, selected_for_vlm=True)
            for c in chosen]
    if cfg.easy_negatives:
        taken = labeled | {(c.a_id, c.b_id) for c in chosen}
        negatives = sample_easy_negatives(universe, emb_a, emb_b, cfg.easy_negatives, taken, cfg.seed,
                                          exclude_a=set(cfg.exclude_a_ids))
        n_check = math.ceil(len(negatives) * cfg.easy_negative_check_share)
        rows += [dict(a_id=c.a_id, b_id=c.b_id, origin=c.origin, rank=c.rank, cosine=c.cosine,
                      selected_for_vlm=i < n_check) for i, c in enumerate(negatives)]
    unembeddable = [f"a:{i}" for i, v in zip(universe.items_a, emb_a) if not v.any()] + \
                   [f"b:{i}" for i, v in zip(universe.items_b, emb_b) if not v.any()]
    config = {**asdict(cfg), "unembeddable_items": unembeddable}
    iteration_id = store.create_iteration(profile.name, version, config, encoder.version)
    store.add_iteration_pairs(iteration_id, rows)
    return iteration_id


def run_iteration(iteration_id: int, store: Store, labeler: Labeler, concurrency: int = 4,
                  progress: Callable[[int, int], None] | None = None) -> None:
    """Label selected pairs that are pending or failed; safe to call again after a crash."""
    iteration = store.get_iteration(iteration_id)
    if labeler.profile_version != iteration["profile_version"]:
        raise ValueError(f"labeler profile version {labeler.profile_version} != iteration profile version "
                         f"{iteration['profile_version']}")
    profile = iteration["profile"]
    if profile_hash(labeler.profile) != store.profile_hash_at(profile, iteration["profile_version"]):
        raise ValueError(f"labeler profile differs from profile {profile!r} version {iteration['profile_version']} "
                         "the iteration was planned with")
    planned_model = iteration["config"].get("labeler_model")
    if planned_model and labeler.vlm.model != planned_model:
        raise ValueError(f"labeler model {labeler.vlm.model!r} != planned model {planned_model!r}")
    pairs = store.iteration_pairs(iteration_id)
    todo = pairs[(pairs["selected_for_vlm"] == 1) & pairs["status"].isin(["pending", "error"])]
    total, done = len(todo), 0

    def work(a_id: str, b_id: str) -> str:
        a, b = store.get_item(profile, "a", a_id), store.get_item(profile, "b", b_id)
        if a is None or b is None:
            return "error"
        return "done" if labeler.label_pair(a, b).error is None else "error"

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(work, r.a_id, r.b_id): (r.a_id, r.b_id) for r in todo.itertuples()}
        for fut in as_completed(futures):
            a_id, b_id = futures[fut]
            try:
                status = fut.result()
            except Exception:
                status = "error"
            store.set_iteration_pair_status(iteration_id, a_id, b_id, status)
            done += 1
            if progress:
                progress(done, total)


def _labels_for(store: Store, profile: str, pairs: pd.DataFrame, model: str, version: int) -> pd.DataFrame:
    vlm = store.query_labels(profile, source="vlm", model=model, profile_version=version)
    vlm = vlm[vlm["error"].isna()].drop_duplicates(["a_id", "b_id"], keep="last")[["a_id", "b_id", "cls"]]
    return pairs.merge(vlm, on=["a_id", "b_id"], how="left")


def _vlm_pairs_count(store: Store, profile: str) -> int:
    total = 0
    for it_id in store.list_iterations(profile)["id"]:
        p = store.iteration_pairs(int(it_id))
        total += int(((p["selected_for_vlm"] == 1) & p["status"].isin(["done", "error"])).sum())
    return total


def stop_triggered(stop: dict | None, store: Store, profile: str, report: dict) -> bool:
    if not stop:
        return False
    kind, value = stop["kind"], int(stop["value"])
    if kind == "max_iterations":
        its = store.list_iterations(profile)
        return int((its["status"] == "done").sum()) >= value
    if kind == "pair_budget":
        return _vlm_pairs_count(store, profile) >= value
    if kind == "min_new_signal":
        if report.get("threshold_missing"):
            return False
        signal = sum(s.get("positives", 0) + s.get("disagreements", 0) for s in report["origin_stats"].values())
        return signal < value
    raise ValueError(f"unknown stop criterion {kind!r}")


def _delta(current: dict, previous: dict | None) -> dict:
    if not previous:
        return {}
    out = {}
    for origin, cur in current.items():
        prev = previous.get(origin, {})
        out[origin] = {k: cur[k] - prev.get(k, 0) for k in ("n", "positives", "disagreements") if k in cur}
    return out


def finalize_iteration(iteration_id: int, store: Store) -> dict:
    iteration = store.get_iteration(iteration_id)
    cfg, profile, version = iteration["config"], iteration["profile"], iteration["profile_version"]
    threshold, model = cfg.get("positive_threshold"), cfg["labeler_model"]
    pairs = store.iteration_pairs(iteration_id)
    sel = _labels_for(store, profile, pairs[pairs["selected_for_vlm"] == 1], model, version)
    report: dict = {"threshold_missing": threshold is None,
                    "origin_stats": origin_stats(sel[sel["origin"] != "easy_negative"], threshold)}

    negs = sel[(sel["origin"] == "easy_negative") & sel["cls"].notna()]
    report["easy_negative_noise"] = (float((negs["cls"] >= threshold).mean()) if threshold is not None and len(negs) else None)

    previous = store.list_iterations(profile)
    previous = previous[(previous["id"] < iteration_id) & (previous["status"] == "done")]
    prev_report = store.get_iteration(int(previous["id"].iloc[-1]))["report"] if len(previous) else None
    report["delta"] = _delta(report["origin_stats"], (prev_report or {}).get("origin_stats"))

    manual = store.manual_test_pairs(profile)
    report["manual_test"] = None
    if manual:
        rows = []
        for a, b, _ in manual:
            human = store.latest_human_label(profile, a, b)
            vlm = store.get_vlm_label(profile, a, b, model, version)
            rows.append({"human": human.cls if human else None, "vlm": vlm.cls if vlm else None})
        df = pd.DataFrame(rows)
        if df["human"].notna().any():
            report["manual_test"] = labeler_vs_human(df, threshold)

    if threshold is not None:
        dis = [(r.a_id, r.b_id) for r in sel.itertuples() if is_disagreement(r.origin, r.cls, threshold)]
        random.Random(cfg.get("seed", 0)).shuffle(dis)
        if dis:
            store.queue_put(f"disagreements:{iteration_id}", profile, dis[: cfg.get("review_size", 50)])

    store.finish_iteration(iteration_id, "done", report)
    stop = cfg.get("stop")
    report["stop"] = {**stop, "triggered": stop_triggered(stop, store, profile, report)} if stop else None
    store.finish_iteration(iteration_id, "done", report)
    return report
