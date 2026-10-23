"""Prometheus metrics of the app (scraped by prometheus from :9001)."""
from prometheus_client import Counter, Gauge, Histogram

PL_VLM_CALLS = Counter("pl_vlm_calls_total", "VLM calls by labeler step and status", ["step", "status"])
PL_VLM_LATENCY = Histogram("pl_vlm_latency_seconds", "VLM call latency by labeler step", ["step"],
                           buckets=(1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144))
PL_ITERATION_PROGRESS = Gauge("pl_iteration_progress_ratio", "Share of selected pairs processed", ["iteration"])


def record_vlm_call(step: str, status: str, latency_s: float) -> None:
    PL_VLM_CALLS.labels(step=step, status=status).inc()
    PL_VLM_LATENCY.labels(step=step).observe(latency_s)
