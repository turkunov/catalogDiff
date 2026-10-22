#!/usr/bin/env python3
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import gradio as gr
import pandas as pd
import requests
from openai import OpenAI
from prometheus_client import Counter, Gauge, Histogram, start_http_server

import run_qwen3_vl_batch as batch


DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-27B")
DEFAULT_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
POOL_NAMES = [p.strip() for p in os.getenv("POOL_NAMES", "pool-a,pool-b,pool-c").split(",") if p.strip()]
RUNS_DIR = os.getenv("LABELING_RUNS_DIR", "labeling_runs")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9001"))
VALID_URL_PREFIXES = ("http://", "https://", "data:", "file://")


JOBS_STARTED = Counter("vlm_jobs_started_total", "Total started labeling jobs", ["pool"])
JOBS_FINISHED = Counter("vlm_jobs_finished_total", "Total finished labeling jobs", ["pool", "status"])
PAIRS_PROCESSED = Counter("vlm_pairs_processed_total", "Total processed pairs", ["pool", "status"])
PAIR_LATENCY = Histogram(
    "vlm_pair_latency_seconds",
    "Latency per processed pair",
    ["pool"],
    buckets=(1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233),
)
POOL_PROGRESS = Gauge("vlm_pool_progress_percent", "Current pool progress in percent", ["pool"])
POOL_ETA_SECONDS = Gauge("vlm_pool_eta_seconds", "Estimated remaining time in seconds", ["pool"])
POOL_SPEED = Gauge("vlm_pool_speed_pairs_per_sec", "Current speed in pairs per second", ["pool"])
POOL_INFLIGHT = Gauge("vlm_pool_inflight", "Pool is actively processing a job (1/0)", ["pool"])
POOL_SUCCESS = Gauge("vlm_pool_success_total", "Success count for active/last job", ["pool"])
POOL_FAILED = Gauge("vlm_pool_failed_total", "Failure count for active/last job", ["pool"])


@dataclass
class JobState:
    running: bool = False
    stop_requested: bool = False
    started_at: float | None = None
    finished_at: float | None = None
    total: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0
    avg_latency_s: float = 0.0
    speed_pairs_per_sec: float = 0.0
    eta_seconds: float = 0.0
    progress_percent: float = 0.0
    model: str = DEFAULT_MODEL
    output_path: str = ""
    last_error: str = ""
    last_status: str = "Idle"
    strict_prompt: str = batch.STRICT_PROMPT_TEMPLATE
    match_prompt: str = batch.MATCH_PROMPT_TEMPLATE
    successful_pairs: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


POOL_STATES: dict[str, JobState] = {name: JobState() for name in POOL_NAMES}


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in batch.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_media_url(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("www."):
        return "https://" + text
    return text


def validate_image_urls(df: pd.DataFrame) -> pd.DataFrame:
    checked = df.copy()
    checked["q_image_url"] = checked["q_image_url"].map(normalize_media_url)
    checked["d_image_url"] = checked["d_image_url"].map(normalize_media_url)

    bad_rows: list[str] = []
    valid_mask: list[bool] = []
    for i, row in checked.iterrows():
        q_ok = row["q_image_url"].startswith(VALID_URL_PREFIXES)
        d_ok = row["d_image_url"].startswith(VALID_URL_PREFIXES)
        is_valid = q_ok and d_ok
        valid_mask.append(is_valid)
        if not is_valid:
            bad_rows.append(
                f"row={i} q_item_id={batch.safe_text(row['q_item_id'])} d_item_id={batch.safe_text(row['d_item_id'])} "
                f"q_image_url='{row['q_image_url']}' d_image_url='{row['d_image_url']}'"
            )

    if bad_rows:
        filtered = checked[pd.Series(valid_mask, index=checked.index)].copy()
        if len(filtered) == 0:
            preview = "\n".join(bad_rows[:10])
            raise ValueError(
                "All rows have missing/invalid image URLs. Allowed schemes: http://, https://, data:, file://.\n"
                f"Bad rows: {len(bad_rows)} / {len(checked)}\n"
                f"Examples:\n{preview}"
            )
        logging.warning(
            "Dropping rows with invalid image URLs: dropped=%d kept=%d",
            len(bad_rows),
            len(filtered),
        )
        return filtered

    return checked


def filter_reachable_image_urls(df: pd.DataFrame, timeout_sec: int = 8) -> pd.DataFrame:
    cache: dict[str, bool] = {}

    def is_reachable(url: str) -> bool:
        if not isinstance(url, str) or not url:
            return False
        if url.startswith(("data:", "file://")):
            return True
        if url in cache:
            return cache[url]
        try:
            resp = requests.get(url, timeout=timeout_sec, stream=True)
            ok = resp.status_code < 400
        except Exception:
            ok = False
        cache[url] = ok
        return ok

    good_mask = (
        df["q_image_url"].map(is_reachable)
        & df["d_image_url"].map(is_reachable)
    )
    filtered = df[good_mask].copy()
    dropped = len(df) - len(filtered)
    if dropped > 0:
        logging.warning(
            "Dropping rows with unreachable image URLs: dropped=%d kept=%d",
            dropped,
            len(filtered),
        )
    if filtered.empty:
        raise ValueError(
            "No rows left after URL reachability check: image links return HTTP errors (e.g. 404) "
            "or are inaccessible from vLLM container."
        )
    return filtered


def parse_manual_pairs_json(raw: str) -> pd.DataFrame:
    if not raw or not raw.strip():
        return pd.DataFrame(columns=batch.REQUIRED_COLUMNS)
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Manual JSON must be an object or array of objects.")
    df = pd.DataFrame(payload)
    ensure_required_columns(df)
    return df


def build_df_from_inputs(
    dataset_file: str | None,
    dataset_path: str,
    manual_pairs_json: str,
    q_item_id: str,
    d_item_id: str,
    q_title: str,
    d_title: str,
    q_description: str,
    d_description: str,
    q_key_params: str,
    d_key_params: str,
    q_image_url: str,
    d_image_url: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    resolved_dataset_path = ""
    if isinstance(dataset_file, str):
        resolved_dataset_path = dataset_file
    elif isinstance(dataset_file, dict):
        # Compatibility fallback for Gradio payloads that pass file metadata dicts.
        resolved_dataset_path = str(dataset_file.get("path") or "")
    resolved_dataset_path = resolved_dataset_path or dataset_path.strip()
    if resolved_dataset_path:
        file_df = batch.load_dataframe(resolved_dataset_path)
        ensure_required_columns(file_df)
        frames.append(file_df[batch.REQUIRED_COLUMNS].copy())

    manual_df = parse_manual_pairs_json(manual_pairs_json)
    if not manual_df.empty:
        frames.append(manual_df[batch.REQUIRED_COLUMNS].copy())

    if q_image_url.strip() and d_image_url.strip():
        one_pair = pd.DataFrame(
            [
                {
                    "q_item_id": q_item_id or "manual-q",
                    "d_item_id": d_item_id or "manual-d",
                    "q_title": q_title,
                    "d_title": d_title,
                    "q_description": q_description,
                    "d_description": d_description,
                    "q_key_params": q_key_params,
                    "d_key_params": d_key_params,
                    "q_image_url": q_image_url,
                    "d_image_url": d_image_url,
                }
            ]
        )
        frames.append(one_pair)

    if not frames:
        raise ValueError("Provide either a dataset file, manual JSON pairs, or a single pair form.")

    df = pd.concat(frames, ignore_index=True)
    ensure_required_columns(df)
    df = validate_image_urls(df)
    return filter_reachable_image_urls(df)


def build_matching_messages(
    row: pd.Series,
    category: str,
    key_params_list: str,
    vis_params_1: dict[str, Any],
    vis_params_2: dict[str, Any],
    match_prompt_template: str,
) -> list[dict[str, Any]]:
    task_text = batch.build_task_text_with_vis(
        row=row,
        category=category,
        key_params_list=key_params_list,
        vis_params_1=vis_params_1,
        vis_params_2=vis_params_2,
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": batch.safe_text(row["q_image_url"])}},
                {"type": "image_url", "image_url": {"url": batch.safe_text(row["d_image_url"])}},
                {"type": "text", "text": match_prompt_template + "\n\n" + task_text},
            ],
        }
    ]


def infer_two_stage_custom(
    client: OpenAI,
    row: pd.Series,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    category: str,
    key_params_list: str,
    strict_prompt_template: str,
    match_prompt_template: str,
) -> tuple[str | None, str | None, dict[str, Any], dict[str, Any], float, float]:
    strict_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": batch.safe_text(row["q_image_url"])}},
                {"type": "image_url", "image_url": {"url": batch.safe_text(row["d_image_url"])}},
                {"type": "text", "text": strict_prompt_template},
            ],
        }
    ]

    strict_start = time.time()
    strict_resp = client.chat.completions.create(
        model=model,
        messages=strict_messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    strict_latency_s = time.time() - strict_start
    strict_text = strict_resp.choices[0].message.content or ""
    strict_text = strict_text.replace("True", "true").replace("False", "false").replace("None", "null")
    vis_params_1 = batch.load_vis_params(strict_text, "товар_1")
    vis_params_2 = batch.load_vis_params(strict_text, "товар_2")

    matching_messages = build_matching_messages(
        row=row,
        category=category,
        key_params_list=key_params_list,
        vis_params_1=vis_params_1,
        vis_params_2=vis_params_2,
        match_prompt_template=match_prompt_template,
    )
    match_start = time.time()
    match_resp = batch.create_completion_with_json_mode_fallback(
        client=client,
        model=model,
        messages=matching_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    match_latency_s = time.time() - match_start
    return (
        strict_text,
        match_resp.choices[0].message.content,
        vis_params_1,
        vis_params_2,
        strict_latency_s,
        match_latency_s,
    )


def reset_state(pool_name: str, model: str, output_path: str, strict_prompt: str, match_prompt: str, total: int) -> None:
    state = POOL_STATES[pool_name]
    with state.lock:
        state.running = True
        state.stop_requested = False
        state.started_at = time.time()
        state.finished_at = None
        state.total = total
        state.processed = 0
        state.success = 0
        state.failed = 0
        state.avg_latency_s = 0.0
        state.speed_pairs_per_sec = 0.0
        state.eta_seconds = 0.0
        state.progress_percent = 0.0
        state.model = model
        state.output_path = output_path
        state.last_error = ""
        state.last_status = "Running"
        state.strict_prompt = strict_prompt
        state.match_prompt = match_prompt
        state.successful_pairs = []


def update_state_after_record(pool_name: str, latency_s: float, ok: bool, pair_record: dict[str, Any] | None = None, error: str = "") -> None:
    state = POOL_STATES[pool_name]
    with state.lock:
        state.processed += 1
        if ok:
            state.success += 1
            if pair_record:
                state.successful_pairs.append(pair_record)
        else:
            state.failed += 1
            state.last_error = error

        elapsed = max(time.time() - (state.started_at or time.time()), 1e-6)
        state.speed_pairs_per_sec = state.processed / elapsed
        state.avg_latency_s = elapsed / max(state.processed, 1)
        remaining = max(state.total - state.processed, 0)
        state.eta_seconds = remaining / state.speed_pairs_per_sec if state.speed_pairs_per_sec > 0 else 0.0
        state.progress_percent = (state.processed / state.total) * 100 if state.total else 0.0

        POOL_PROGRESS.labels(pool=pool_name).set(state.progress_percent)
        POOL_ETA_SECONDS.labels(pool=pool_name).set(state.eta_seconds)
        POOL_SPEED.labels(pool=pool_name).set(state.speed_pairs_per_sec)
        POOL_SUCCESS.labels(pool=pool_name).set(state.success)
        POOL_FAILED.labels(pool=pool_name).set(state.failed)


def finish_state(pool_name: str, status: str) -> None:
    state = POOL_STATES[pool_name]
    with state.lock:
        state.running = False
        state.finished_at = time.time()
        state.last_status = status
        POOL_INFLIGHT.labels(pool=pool_name).set(0)


def worker_run(
    pool_name: str,
    df: pd.DataFrame,
    model: str,
    base_url: str,
    api_key: str,
    output_path: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int,
    category: str,
    key_params_list: str,
    strict_prompt: str,
    match_prompt: str,
) -> None:
    state = POOL_STATES[pool_name]
    status = "Completed"
    try:
        batch.check_health(base_url, timeout_sec=20)
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=600)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for idx, row in df.reset_index(drop=True).iterrows():
            with state.lock:
                if state.stop_requested:
                    status = "Stopped by user"
                    break

            q_id = batch.safe_text(row["q_item_id"])
            d_id = batch.safe_text(row["d_item_id"])
            row_start = time.time()
            error_msg = ""
            response_text = None
            strict_response_text = None
            vis_params_1: dict[str, Any] = {}
            vis_params_2: dict[str, Any] = {}
            strict_latency_s = None
            match_latency_s = None

            for attempt in range(1, retries + 2):
                try:
                    (
                        strict_response_text,
                        response_text,
                        vis_params_1,
                        vis_params_2,
                        strict_latency_s,
                        match_latency_s,
                    ) = infer_two_stage_custom(
                        client=client,
                        row=row,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        category=category,
                        key_params_list=key_params_list,
                        strict_prompt_template=strict_prompt,
                        match_prompt_template=match_prompt,
                    )
                    break
                except Exception as exc:
                    error_msg = str(exc)
                    if attempt <= retries:
                        time.sleep(1.5)

            latency_s = time.time() - row_start
            pair_record = {
                "q_item_id": q_id,
                "d_item_id": d_id,
                "q_image_url": batch.safe_text(row["q_image_url"]),
                "d_image_url": batch.safe_text(row["d_image_url"]),
                "predicted_class": batch.extract_predicted_class(response_text),
                "latency_total_s": latency_s,
            }
            output_record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pool_name": pool_name,
                "row_index": int(idx),
                "q_item_id": q_id,
                "d_item_id": d_id,
                "q_title": batch.safe_text(row["q_title"]),
                "d_title": batch.safe_text(row["d_title"]),
                "q_key_params": batch.safe_text(row["q_key_params"]),
                "d_key_params": batch.safe_text(row["d_key_params"]),
                "q_image_url": batch.safe_text(row["q_image_url"]),
                "d_image_url": batch.safe_text(row["d_image_url"]),
                "strict_response_text": strict_response_text,
                "response_text": response_text,
                "vis_params_1": vis_params_1,
                "vis_params_2": vis_params_2,
                "latency_strict_s": round(strict_latency_s, 4) if strict_latency_s else None,
                "latency_match_s": round(match_latency_s, 4) if match_latency_s else None,
                "latency_total_s": round(latency_s, 4),
                "error": error_msg or None,
            }
            batch.write_jsonl(output_path, output_record)

            ok = response_text is not None
            update_state_after_record(
                pool_name=pool_name,
                latency_s=latency_s,
                ok=ok,
                pair_record=pair_record if ok else None,
                error=error_msg,
            )
            PAIRS_PROCESSED.labels(pool=pool_name, status="success" if ok else "failed").inc()
            PAIR_LATENCY.labels(pool=pool_name).observe(latency_s)

    except Exception as exc:
        status = f"Failed: {exc}"
    finally:
        finish_state(pool_name, status=status)
        JOBS_FINISHED.labels(pool=pool_name, status="success" if status.startswith("Completed") else "error").inc()


def format_status(pool_name: str) -> tuple[str, float, str]:
    state = POOL_STATES[pool_name]
    with state.lock:
        started = datetime.fromtimestamp(state.started_at).isoformat() if state.started_at else "-"
        eta = f"{state.eta_seconds:.1f}s"
        speed = f"{state.speed_pairs_per_sec:.3f} pair/s"
        runtime = f"{(time.time() - state.started_at):.1f}s" if state.started_at else "0.0s"
        status_lines = [
            f"**Pool:** `{pool_name}`",
            f"**Status:** {state.last_status}",
            f"**Model:** `{state.model}`",
            f"**Progress:** {state.processed}/{state.total} ({state.progress_percent:.2f}%)",
            f"**Success / Failed:** {state.success} / {state.failed}",
            f"**Speed:** {speed}",
            f"**ETA:** {eta}",
            f"**Average latency:** {state.avg_latency_s:.3f}s",
            f"**Runtime:** {runtime}",
            f"**Output:** `{state.output_path or '-'}`",
            f"**Started at:** {started}",
        ]
        if state.last_error:
            status_lines.append(f"**Last error:** `{state.last_error}`")
        return "\n\n".join(status_lines), state.progress_percent, state.output_path


def stop_job(pool_name: str) -> str:
    state = POOL_STATES[pool_name]
    with state.lock:
        if not state.running:
            return f"Pool `{pool_name}` is not running."
        state.stop_requested = True
    return f"Stop requested for `{pool_name}`."


def start_job(
    pool_name: str,
    dataset_file: str | None,
    dataset_path: str,
    manual_pairs_json: str,
    q_item_id: str,
    d_item_id: str,
    q_title: str,
    d_title: str,
    q_description: str,
    d_description: str,
    q_key_params: str,
    d_key_params: str,
    q_image_url: str,
    d_image_url: str,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int,
    category: str,
    key_params_list: str,
    strict_prompt: str,
    match_prompt: str,
) -> tuple[str, float]:
    state = POOL_STATES[pool_name]
    with state.lock:
        if state.running:
            return f"Pool `{pool_name}` already has a running job.", state.progress_percent

    uploaded_dataset_path = dataset_file if isinstance(dataset_file, str) else dataset_file
    df = build_df_from_inputs(
        dataset_file=uploaded_dataset_path,
        dataset_path=dataset_path,
        manual_pairs_json=manual_pairs_json,
        q_item_id=q_item_id,
        d_item_id=d_item_id,
        q_title=q_title,
        d_title=d_title,
        q_description=q_description,
        d_description=d_description,
        q_key_params=q_key_params,
        d_key_params=d_key_params,
        q_image_url=q_image_url,
        d_image_url=d_image_url,
    )

    os.makedirs(RUNS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = os.path.join(RUNS_DIR, f"{pool_name}_{timestamp}.jsonl")

    reset_state(
        pool_name=pool_name,
        model=model,
        output_path=output_path,
        strict_prompt=strict_prompt,
        match_prompt=match_prompt,
        total=len(df),
    )
    JOBS_STARTED.labels(pool=pool_name).inc()
    POOL_INFLIGHT.labels(pool=pool_name).set(1)

    thread = threading.Thread(
        target=worker_run,
        kwargs={
            "pool_name": pool_name,
            "df": df,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "output_path": output_path,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "retries": retries,
            "category": category,
            "key_params_list": key_params_list,
            "strict_prompt": strict_prompt,
            "match_prompt": match_prompt,
        },
        daemon=True,
    )
    thread.start()

    status_md, progress, _ = format_status(pool_name)
    return status_md, progress


def render_sample(pool_name: str, sample_size: int) -> tuple[str, str | None]:
    state = POOL_STATES[pool_name]
    with state.lock:
        if not state.successful_pairs:
            return "No successful labeled pairs yet.", None
        pairs = list(state.successful_pairs)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sample_dir = os.path.join(RUNS_DIR, "sample_pair_plots")
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, f"{pool_name}_{ts}.png")
    batch.plot_random_pairs_grid(
        pairs=pairs,
        save_path=sample_path,
        n_rows=max(1, sample_size),
    )
    return f"Sample rendered from {len(pairs)} labeled pairs.", sample_path


def create_pool_tab(pool_name: str) -> None:
    def start_for_pool(
        dataset_file: str | None,
        dataset_path: str,
        manual_pairs_json: str,
        q_item_id: str,
        d_item_id: str,
        q_title: str,
        d_title: str,
        q_description: str,
        d_description: str,
        q_key_params: str,
        d_key_params: str,
        q_image_url: str,
        d_image_url: str,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: float,
        temperature: float,
        top_p: float,
        retries: float,
        category: str,
        key_params_list: str,
        strict_prompt: str,
        match_prompt: str,
    ) -> tuple[str, float]:
        return start_job(
            pool_name=pool_name,
            dataset_file=dataset_file,
            dataset_path=dataset_path,
            manual_pairs_json=manual_pairs_json,
            q_item_id=q_item_id,
            d_item_id=d_item_id,
            q_title=q_title,
            d_title=d_title,
            q_description=q_description,
            d_description=d_description,
            q_key_params=q_key_params,
            d_key_params=d_key_params,
            q_image_url=q_image_url,
            d_image_url=d_image_url,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p,
            retries=int(retries),
            category=category,
            key_params_list=key_params_list,
            strict_prompt=strict_prompt,
            match_prompt=match_prompt,
        )

    with gr.Tab(pool_name):
        gr.Markdown(f"### Pool: `{pool_name}`")
        status_md = gr.Markdown("Idle")
        progress = gr.Slider(label="Progress (%)", minimum=0, maximum=100, value=0, interactive=False)
        output_path_box = gr.Textbox(label="Output JSONL path", value="", interactive=False)

        with gr.Row():
            model = gr.Textbox(label="Model", value=DEFAULT_MODEL)
            base_url = gr.Textbox(label="vLLM Base URL", value=DEFAULT_BASE_URL)
            api_key = gr.Textbox(label="API Key", value=DEFAULT_API_KEY, type="password")

        with gr.Row():
            max_tokens = gr.Number(label="Max tokens", value=1024, precision=0)
            temperature = gr.Number(label="Temperature", value=0.1)
            top_p = gr.Number(label="Top P", value=0.3)
            retries = gr.Number(label="Retries", value=2, precision=0)

        with gr.Row():
            category = gr.Textbox(label="Category", value="Диваны")
            key_params_list = gr.Textbox(
                label="Key params list",
                value="материал обивки, форма, основной цвет",
            )

        dataset_file = gr.File(
            label="Dataset file upload (.csv/.parquet/.pkl/.feather)",
            type="filepath",
            file_types=[".csv", ".parquet", ".pkl", ".pickle", ".feather"],
        )
        dataset_path = gr.Textbox(
            label="Dataset path on server/container (optional, avoids browser upload)",
            placeholder="/workspace/notebooks/your_dataset.csv",
        )
        manual_pairs_json = gr.Textbox(
            label="Manual pairs JSON (object or array of objects)",
            placeholder='[{"q_item_id":"1","d_item_id":"2","q_title":"...","d_title":"...","q_description":"","d_description":"","q_key_params":"{}","d_key_params":"{}","q_image_url":"https://...","d_image_url":"https://..."}]',
            lines=6,
        )

        with gr.Accordion("Single pair quick input (optional)", open=False):
            with gr.Row():
                q_item_id = gr.Textbox(label="q_item_id")
                d_item_id = gr.Textbox(label="d_item_id")
            with gr.Row():
                q_title = gr.Textbox(label="q_title")
                d_title = gr.Textbox(label="d_title")
            with gr.Row():
                q_description = gr.Textbox(label="q_description")
                d_description = gr.Textbox(label="d_description")
            with gr.Row():
                q_key_params = gr.Textbox(label="q_key_params", value="{}")
                d_key_params = gr.Textbox(label="d_key_params", value="{}")
            with gr.Row():
                q_image_url = gr.Textbox(label="q_image_url")
                d_image_url = gr.Textbox(label="d_image_url")

        with gr.Accordion("Prompt templates", open=False):
            strict_prompt = gr.Textbox(label="STRICT prompt template", value=batch.STRICT_PROMPT_TEMPLATE, lines=14)
            match_prompt = gr.Textbox(label="MATCH prompt template", value=batch.MATCH_PROMPT_TEMPLATE, lines=18)

        with gr.Row():
            start_btn = gr.Button("Start labeling", variant="primary")
            stop_btn = gr.Button("Stop labeling", variant="stop")

        stop_msg = gr.Markdown("")
        with gr.Row():
            sample_size = gr.Number(label="Sample rows", value=6, precision=0)
            sample_btn = gr.Button("Render labeled sample grid")
        sample_info = gr.Markdown("")
        sample_image = gr.Image(label="Labeled sample", type="filepath")

        timer = gr.Timer(1.0)
        timer.tick(
            fn=lambda: format_status(pool_name),
            inputs=None,
            outputs=[status_md, progress, output_path_box],
        )

        start_btn.click(
            fn=start_for_pool,
            inputs=[
                dataset_file,
                dataset_path,
                manual_pairs_json,
                q_item_id,
                d_item_id,
                q_title,
                d_title,
                q_description,
                d_description,
                q_key_params,
                d_key_params,
                q_image_url,
                d_image_url,
                model,
                base_url,
                api_key,
                max_tokens,
                temperature,
                top_p,
                retries,
                category,
                key_params_list,
                strict_prompt,
                match_prompt,
            ],
            outputs=[status_md, progress],
        )

        stop_btn.click(
            fn=lambda: stop_job(pool_name),
            inputs=None,
            outputs=stop_msg,
        )

        sample_btn.click(
            fn=lambda n: render_sample(pool_name, int(n)),
            inputs=sample_size,
            outputs=[sample_info, sample_image],
        )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Qwen3-VL Labeling Control Center") as demo:
        gr.Markdown(
            "# Qwen3-VL Labeling Demo\n"
            "Run multiple labeling jobs in parallel pools with live progress, ETA, speed, and sample visualization."
        )
        for pool_name in POOL_NAMES:
            create_pool_tab(pool_name)
    return demo


def main() -> None:
    batch.setup_logger()
    start_http_server(METRICS_PORT)
    app = build_app()
    app.queue(default_concurrency_limit=16).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False,
        max_file_size="5gb",
    )


if __name__ == "__main__":
    main()
