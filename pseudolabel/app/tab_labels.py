"""Labels tab: browse the label store with filters, open a pair card, export the selection."""
from dataclasses import asdict
from datetime import datetime, timezone

import gradio as gr
import pandas as pd

from pseudolabel.app.context import AppContext

FILTERS = ("iteration_id", "origin", "cls", "model", "profile_version", "source", "disagreement")


def search_labels(ctx: AppContext, profile: str, filters: dict) -> pd.DataFrame:
    clean = {k: v for k, v in filters.items() if k in FILTERS and v not in (None, "", False)}
    for key in ("iteration_id", "cls", "profile_version"):
        if key in clean:
            clean[key] = int(clean[key])
    return ctx.store.query_labels(profile, **clean)


def pair_card(ctx: AppContext, profile: str, a_id: str, b_id: str) -> dict:
    a, b = ctx.store.get_item(profile, "a", a_id), ctx.store.get_item(profile, "b", b_id)
    labels = ctx.store.query_labels(profile)
    labels = labels[(labels["a_id"] == a_id) & (labels["b_id"] == b_id)]
    return {
        "a": asdict(a) if a else None,
        "b": asdict(b) if b else None,
        "annotations": {"a": ctx.store.annotations(profile, "a", a_id), "b": ctx.store.annotations(profile, "b", b_id)},
        "labels": labels.drop(columns=["origin", "rank", "cosine", "iteration_id"]).to_dict("records"),
        "origins": ctx.store.pair_origins(profile, a_id, b_id).to_dict("records"),
    }


def export_selection(ctx: AppContext, df: pd.DataFrame) -> str:
    out = ctx.settings.work_dir / "exports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"labels_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    df.to_csv(path, index=False)
    return str(path)


def build(ctx: AppContext) -> None:
    names = ctx.profile_names()
    with gr.Row():
        profile = gr.Dropdown(names, value="sofas" if "sofas" in names else (names[0] if names else None), label="Profile")
        iteration = gr.Number(label="Iteration", precision=0, value=None)
        origin = gr.Dropdown(["", "top_1_10", "top_11_100", "top_101_1000", "title", "random", "easy_negative"], value="", label="Origin")
        cls = gr.Dropdown(["", "1", "2", "3", "4", "5"], value="", label="Class")
        source = gr.Dropdown(["", "vlm", "human", "legacy"], value="", label="Source")
    with gr.Row():
        model = gr.Textbox(label="Model")
        version = gr.Number(label="Profile version", precision=0, value=None)
        disagreement = gr.Checkbox(label="Only queued disagreements")
        search = gr.Button("Search", variant="primary")
        export = gr.Button("Export selection")
    table = gr.Dataframe(interactive=False, wrap=True)
    with gr.Row():
        a_id, b_id = gr.Textbox(label="Listing 1 id"), gr.Textbox(label="Listing 2 id")
        open_card = gr.Button("Open pair")
    with gr.Row():
        a_img, b_img = gr.Image(label="Listing 1", interactive=False, height=320), gr.Image(label="Listing 2", interactive=False, height=320)
    card = gr.JSON(label="Pair card")
    exported = gr.File(label="Export")
    current = gr.State(pd.DataFrame())

    def on_search(p, it, o, c, s, m, v, d):
        df = search_labels(ctx, p, {"iteration_id": it, "origin": o, "cls": c, "source": s, "model": m,
                                    "profile_version": v, "disagreement": d})
        return df, df

    def on_open(p, a, b):
        data = pair_card(ctx, p, a, b)
        return (data["a"] or {}).get("image"), (data["b"] or {}).get("image"), data

    search.click(on_search, [profile, iteration, origin, cls, source, model, version, disagreement], [table, current])
    export.click(lambda df: export_selection(ctx, df), current, exported)
    open_card.click(on_open, [profile, a_id, b_id], [a_img, b_img, card])
