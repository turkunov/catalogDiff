"""Entry point: python -m pseudolabel.app.main (Gradio on :7860, Prometheus metrics on :9001)."""
import os

import gradio as gr
from prometheus_client import start_http_server

from pseudolabel.app import tab_pair, tab_profiles
from pseudolabel.app.context import AppContext, make_context
from pseudolabel.settings import load_settings

TABS = ("Profiles", "Pair labeling", "Iterations", "Manual review", "Labels")


def build_app(ctx: AppContext) -> gr.Blocks:
    with gr.Blocks(title="Pseudo-labeling") as demo:
        gr.Markdown("# Pseudo-labeling of listing pairs")
        with gr.Tabs():
            with gr.Tab("Profiles"):
                tab_profiles.build(ctx)
            with gr.Tab("Pair labeling"):
                tab_pair.build(ctx)
            for name in TABS[2:]:
                with gr.Tab(name):
                    gr.Markdown("Coming soon.")
    return demo


def main() -> None:
    settings = load_settings()
    ctx = make_context(settings)
    start_http_server(int(os.getenv("PL_METRICS_PORT", "9001")))
    build_app(ctx).queue(default_concurrency_limit=8).launch(
        server_name="0.0.0.0", server_port=7860, show_api=False,
        allowed_paths=[str(ctx.repo_root / "demo"), "/data"])


if __name__ == "__main__":
    main()
