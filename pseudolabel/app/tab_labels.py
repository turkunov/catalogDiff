"""Labels tab: browse the label store with filters, open a pair card, export the selection."""
from dataclasses import asdict
from datetime import datetime, timezone

import gradio as gr
import pandas as pd

from pseudolabel.app.context import AppContext

FILTERS = ("iteration_id", "origin", "cls", "model", "profile_version", "source", "disagreement")
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
LOW_CONFIDENCE_QUEUE = "low_confidence:{profile}"


def flag_low_confidence(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Marks VLM labels whose most likely class has probability below `threshold` (no strong confidence in one class)."""
    df = df.copy()
    conf = pd.to_numeric(df["confidence"], errors="coerce") if "confidence" in df else pd.Series(dtype=float, index=df.index)
    df["low_confidence"] = (df["source"] == "vlm") & conf.notna() & (conf < threshold)
    return df


def search_labels(ctx: AppContext, profile: str, filters: dict) -> pd.DataFrame:
    """Labels by filters; `confidence_threshold` flags low confidence, `low_confidence_below` keeps only those."""
    clean = {k: v for k, v in filters.items() if k in FILTERS and v not in (None, "", False)}
    for key in ("iteration_id", "cls", "profile_version"):
        if key in clean:
            clean[key] = int(clean[key])
    only_low = filters.get("low_confidence_below")
    threshold = only_low if only_low not in (None, "") else filters.get("confidence_threshold")
    df = flag_low_confidence(ctx.store.query_labels(profile, **clean),
                             float(threshold) if threshold not in (None, "") else DEFAULT_CONFIDENCE_THRESHOLD)
    return df[df["low_confidence"]].reset_index(drop=True) if only_low not in (None, "") else df


def queue_low_confidence(ctx: AppContext, profile: str, df: pd.DataFrame) -> int:
    """Puts the flagged pairs of a search result into the manual review queue low_confidence:<profile>."""
    pairs = list(dict.fromkeys(zip(df.loc[df["low_confidence"], "a_id"], df.loc[df["low_confidence"], "b_id"])))
    ctx.store.queue_put(LOW_CONFIDENCE_QUEUE.format(profile=profile), profile, pairs)
    return len(pairs)


def _highlight(df: pd.DataFrame):
    """Low-confidence rows on a yellow background in the table."""
    if df.empty or "low_confidence" not in df:
        return df
    return df.style.apply(lambda r: ["background-color: #fff3bf" if r["low_confidence"] else "" for _ in r], axis=1)


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
    with gr.Row():
        conf_threshold = gr.Number(label="Low confidence below", value=DEFAULT_CONFIDENCE_THRESHOLD, minimum=0, maximum=1)
        only_low = gr.Checkbox(label="Only low confidence")
        search = gr.Button("Search", variant="primary")
        export = gr.Button("Export selection")
        to_review = gr.Button("Send low-confidence to manual review")
    queued = gr.Markdown()
    table = gr.Dataframe(interactive=False, wrap=True)
    with gr.Row():
        a_id, b_id = gr.Textbox(label="Listing 1 id"), gr.Textbox(label="Listing 2 id")
        open_card = gr.Button("Open pair")
    with gr.Row():
        a_img, b_img = gr.Image(label="Listing 1", interactive=False, height=320), gr.Image(label="Listing 2", interactive=False, height=320)
    card = gr.JSON(label="Pair card")
    exported = gr.File(label="Export")
    current = gr.State(pd.DataFrame())

    def on_search(p, it, o, c, s, m, v, d, t, low):
        df = search_labels(ctx, p, {"iteration_id": it, "origin": o, "cls": c, "source": s, "model": m,
                                    "profile_version": v, "disagreement": d, "confidence_threshold": t,
                                    "low_confidence_below": t if low else None})
        return _highlight(df), df

    def on_queue(p, df):
        if df is None or df.empty:
            return "Search first."
        n = queue_low_confidence(ctx, p, df)
        return f"{n} low-confidence pairs are in the queue `{LOW_CONFIDENCE_QUEUE.format(profile=p)}` (Review tab)."

    def on_open(p, a, b):
        data = pair_card(ctx, p, a, b)
        return (data["a"] or {}).get("image"), (data["b"] or {}).get("image"), data

    search.click(on_search, [profile, iteration, origin, cls, source, model, version, disagreement, conf_threshold, only_low],
                 [table, current])
    to_review.click(on_queue, [profile, current], queued)
    export.click(lambda df: export_selection(ctx, df), current, exported)
    open_card.click(on_open, [profile, a_id, b_id], [a_img, b_img, card])
