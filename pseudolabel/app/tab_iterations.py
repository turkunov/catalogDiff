"""Iterations tab: configure and start an iteration (runs in the background), resume, report table, export."""
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

from pseudolabel.app.context import AppContext
from pseudolabel.app.metrics import PL_ITERATION_PROGRESS
from pseudolabel.data import load_universe
from pseudolabel.export import build_training_set, write_export
from pseudolabel.iteration import IterationConfig, finalize_iteration, labeler_options, plan_iteration, run_iteration
from pseudolabel.metrics import ThresholdRequired
from pseudolabel.sampling import DEFAULT_SHARES

_JOBS: dict[int, threading.Thread] = {}
WEIGHTS_FORMAT = "Weights must look like '3:0.5, 4:0.8'"


def _config(ctx: AppContext, form: dict) -> IterationConfig:
    fields = {k: v for k, v in form.items() if v not in (None, "")}
    stop = {"kind": fields.pop("stop_kind"), "value": int(fields.pop("stop_value"))} if fields.get("stop_kind") and fields.get("stop_value") is not None else None
    fields.pop("stop_kind", None), fields.pop("stop_value", None)
    profile = fields.pop("profile")
    cfg = IterationConfig(profile_name=profile, pairs_path=fields.pop("pairs_path"), mode=fields.pop("mode"), stop=stop,
                          **{k: v for k, v in fields.items() if k in IterationConfig.__dataclass_fields__})
    cfg.labeler_model = cfg.labeler_model or ctx.settings.vllm_model
    for key in ("budget", "easy_negatives", "review_size", "seed"):
        setattr(cfg, key, int(getattr(cfg, key)))
    if cfg.positive_threshold is not None:
        cfg.positive_threshold = int(cfg.positive_threshold)
    return cfg


def _run(ctx: AppContext, iteration_id: int) -> None:
    try:
        _label_and_finalize(ctx, iteration_id)
    except Exception as e:  # a background thread has no caller: record the failure where the UI shows it
        logging.exception("iteration %s failed", iteration_id)
        ctx.store.finish_iteration(iteration_id, "failed", {"error": f"{type(e).__name__}: {e}"})


def _label_and_finalize(ctx: AppContext, iteration_id: int) -> None:
    store = ctx.store
    iteration = store.get_iteration(iteration_id)
    # Label with the profile exactly as planned, even if the YAML file was edited since.
    profile = store.profile_at(iteration["profile"], iteration["profile_version"])
    labeler = ctx.labeler(profile, iteration["profile_version"], iteration["config"].get("labeler_model"),
                          labeler_options(iteration["config"]))
    gauge = PL_ITERATION_PROGRESS.labels(iteration=str(iteration_id))
    run_iteration(iteration_id, store, labeler, ctx.settings.vlm_concurrency,
                  progress=lambda done, total: gauge.set(done / total if total else 1.0))
    finalize_iteration(iteration_id, store)


def _launch(ctx: AppContext, iteration_id: int) -> None:
    thread = threading.Thread(target=_run, args=(ctx, iteration_id), daemon=True, name=f"iteration-{iteration_id}")
    _JOBS[iteration_id] = thread
    thread.start()


def start_iteration(ctx: AppContext, form: dict) -> int:
    """Plan synchronously, then label and finalize in a background thread."""
    cfg = _config(ctx, form)
    profile = ctx.profile(cfg.profile_name)
    universe = load_universe(profile, Path(cfg.pairs_path))
    iteration_id = plan_iteration(cfg, ctx.store, profile, universe, ctx.encoder_factory(cfg.encoder_checkpoint),
                                  ctx.settings.work_dir / "embeddings")
    _launch(ctx, iteration_id)
    return iteration_id


_LAUNCH_LOCK = threading.Lock()


def resume_iteration(ctx: AppContext, iteration_id: int) -> None:
    with _LAUNCH_LOCK:  # two quick Resume clicks must not start two labelers on the same pairs
        if _alive(iteration_id):
            return
        ctx.store.set_iteration_status(iteration_id, "running")
        _launch(ctx, iteration_id)


def wait(iteration_id: int, timeout: float | None = None) -> None:
    if iteration_id in _JOBS:
        _JOBS[iteration_id].join(timeout)


def _alive(iteration_id: int) -> bool:
    return iteration_id in _JOBS and _JOBS[iteration_id].is_alive()


def _eta_minutes(created_at: str, processed: int, pending: int) -> float | None:
    if not processed or not pending:
        return None
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(created_at)).total_seconds()
    return round(elapsed / processed * pending / 60, 1)


def iterations_table(ctx: AppContext, profile: str) -> pd.DataFrame:
    rows = []
    for it in ctx.store.list_iterations(profile).itertuples():
        iteration_id = int(it.id)
        full = ctx.store.get_iteration(iteration_id)
        pairs = ctx.store.iteration_pairs(iteration_id)
        selected = pairs[pairs["selected_for_vlm"] == 1]["status"]
        done, error, pending = (int((selected == s).sum()) for s in ("done", "error", "pending"))
        stats = ((full["report"] or {}).get("origin_stats") or {}).values()
        stop = (full["report"] or {}).get("stop") or {}
        running = it.status == "running" and _alive(iteration_id)
        rows.append({
            "id": iteration_id, "status": "interrupted" if it.status == "running" and not running else it.status,
            "mode": full["config"].get("mode"), "encoder": it.encoder_version, "budget": full["config"].get("budget"),
            "done": done, "error": error, "pending": pending,
            "eta_min": _eta_minutes(it.created_at, done + error, pending) if running else None,
            "positives": sum(s.get("positives", 0) for s in stats),
            "disagreements": sum(s.get("disagreements", 0) for s in stats),
            "stop_triggered": stop.get("triggered"),
        })
    return pd.DataFrame(rows, columns=["id", "status", "mode", "encoder", "budget", "done", "error", "pending", "eta_min",
                                       "positives", "disagreements", "stop_triggered"]).astype({"eta_min": object})


def parse_weights(text: str) -> dict[int, float]:
    weights = {}
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        try:
            cls, w = part.split(":")
            weights[int(cls)] = float(w)
        except ValueError as e:
            raise ValueError(WEIGHTS_FORMAT) from e
    return weights


def export_iteration(ctx: AppContext, profile: str, iteration_ids: list[int] | None, threshold: int | None,
                     weights_text: str) -> tuple[str | None, str]:
    try:
        weights = parse_weights(weights_text)
        df = build_training_set(ctx.store, profile, threshold, weights, iteration_ids or None)
    except (ThresholdRequired, ValueError) as e:
        return None, str(e)
    stem = f"{profile}_threshold{threshold}_" + ("all" if not iteration_ids else "_".join(map(str, iteration_ids)))
    parquet, csv = write_export(df, ctx.settings.work_dir / "exports", stem)
    return str(parquet), f"Exported {len(df)} rows to {parquet} and {csv}"


def build(ctx: AppContext) -> None:
    names = ctx.profile_names()
    with gr.Row():
        profile = gr.Dropdown(names, value="sofas" if "sofas" in names else (names[0] if names else None), label="Profile")
        pairs_path = gr.Textbox(label="Pairs file", value="/data/train_test_with_images/combined_labeled_df_balanced_small.csv")
        mode = gr.Radio(["pairs_file", "encoder_search"], value="pairs_file", label="Pairs come from")
    with gr.Row():
        encoder = gr.Textbox(label="Encoder checkpoint", value="google/siglip2-so400m-patch14-384")
        model = gr.Textbox(label="Labeler model", value=ctx.settings.vllm_model)
        budget = gr.Number(label="Budget (new VLM pairs)", value=200, precision=0)
    with gr.Row():
        shares = [gr.Number(label=f"Share {k}", value=v) for k, v in DEFAULT_SHARES.items()]
        title_share = gr.Number(label="Title share (encoder_search)", value=0.10)
    with gr.Row():
        threshold = gr.Number(label="Positive threshold (empty = not set)", precision=0, value=None)
        easy = gr.Number(label="Easy negatives", value=0, precision=0)
        review = gr.Number(label="Disagreements to review", value=50, precision=0)
        stop_kind = gr.Dropdown(["", "max_iterations", "pair_budget", "min_new_signal"], value="", label="Stop criterion")
        stop_value = gr.Number(label="Stop value", precision=0, value=None)
        seed = gr.Number(label="Seed", value=0, precision=0)
    with gr.Row():
        start = gr.Button("Start", variant="primary")
        iteration_id = gr.Number(label="Iteration id", precision=0, value=None)
        resume = gr.Button("Resume")
        refresh = gr.Button("Refresh")
    message = gr.Markdown()
    table = gr.Dataframe(label="Iterations", interactive=False)
    report = gr.JSON(label="Report of the selected iteration")
    with gr.Row():
        export_ids = gr.Textbox(label="Iterations to export (comma separated, empty = all)")
        weights = gr.Textbox(label="Class weights, e.g. 3:0.5, 4:0.8")
        export = gr.Button("Export")
    export_file = gr.File(label="Export")

    def on_start(p, path, m, enc, mdl, b, s1, s2, s3, s4, ts, t, e, r, sk, sv, sd):
        form = {"profile": p, "pairs_path": path, "mode": m, "encoder_checkpoint": enc, "labeler_model": mdl,
                "budget": b, "shares": dict(zip(DEFAULT_SHARES, (s1, s2, s3, s4))), "title_share": ts,
                "positive_threshold": t, "easy_negatives": e, "review_size": r, "stop_kind": sk, "stop_value": sv, "seed": sd}
        try:
            it = start_iteration(ctx, form)
            return f"Started iteration {it}", it, iterations_table(ctx, p)
        except Exception as ex:
            return f"**Error:** `{ex}`", None, iterations_table(ctx, p)

    def on_refresh(p, it):
        rep = ctx.store.get_iteration(int(it))["report"] if it else None
        return iterations_table(ctx, p), rep

    def on_resume(p, it):
        if not it:
            return "Choose an iteration id", iterations_table(ctx, p)
        resume_iteration(ctx, int(it))
        return f"Resumed iteration {int(it)}", iterations_table(ctx, p)

    def on_export(p, ids, t, w):
        id_list = [int(x) for x in ids.split(",") if x.strip()] if ids else None
        path, msg = export_iteration(ctx, p, id_list, None if t in (None, "") else int(t), w)
        return path, msg

    start.click(on_start, [profile, pairs_path, mode, encoder, model, budget, *shares, title_share, threshold, easy,
                           review, stop_kind, stop_value, seed], [message, iteration_id, table])
    refresh.click(on_refresh, [profile, iteration_id], [table, report])
    resume.click(on_resume, [profile, iteration_id], [message, table])
    export.click(on_export, [profile, export_ids, threshold, weights], [export_file, message])
