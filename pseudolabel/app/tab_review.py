"""Manual review tab: blind 1–5 labeling of a queue (manual test or disagreements) and the manual test runs."""
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from pseudolabel.app.context import AppContext
from pseudolabel.data import load_universe
from pseudolabel.manual_test import import_human_labels, load_manual_test_csv, run_manual_test, select_manual_test
from pseudolabel.store import LabelRow


@dataclass
class ReviewCard:
    """What a reviewer sees: the two listings only — never the VLM class, reasoning, origin, rank or cosine."""

    a_id: str
    b_id: str
    a_image: str
    b_image: str
    a_title: str
    b_title: str
    a_description: str
    b_description: str
    a_params: dict
    b_params: dict
    position: str


def next_card(ctx: AppContext, queue: str) -> ReviewCard | None:
    profile = ctx.store.queue_profile(queue)
    pairs = ctx.store.queue_pairs(queue)
    done = [p for p in pairs if ctx.store.latest_human_label(profile, *p) is not None]
    todo = [p for p in pairs if p not in set(done)]
    if not todo:
        return None
    a_id, b_id = todo[0]
    a, b = ctx.store.get_item(profile, "a", a_id), ctx.store.get_item(profile, "b", b_id)
    return ReviewCard(a_id, b_id, a.image, b.image, a.title, b.title, a.description, b.description,
                      a.key_params, b.key_params, f"{len(done) + 1}/{len(pairs)}")


def submit(ctx: AppContext, queue: str, a_id: str, b_id: str, cls: int, comment: str) -> ReviewCard | None:
    if int(cls) not in range(1, 6):
        raise ValueError("class must be 1..5")
    ctx.store.put_label(LabelRow(ctx.store.queue_profile(queue), a_id, b_id, "human", int(cls), comment=comment or ""))
    return next_card(ctx, queue)


def prepare_manual_test(ctx: AppContext, profile: str, pairs_path: str, per_class: int, from_csv: str | None = None) -> str:
    universe = load_universe(ctx.profile(profile), Path(pairs_path))
    ctx.store.upsert_items(profile, list(universe.items_a.values()) + list(universe.items_b.values()))
    if from_csv:
        rows, shortfall = load_manual_test_csv(Path(from_csv)), {}
    else:
        rows, shortfall = select_manual_test(universe, int(per_class))
    ctx.store.set_manual_test(profile, rows)
    ctx.store.queue_put("manual_test", profile, [(a, b) for a, b, _ in rows])
    return f"Manual test: {len(rows)} pairs" + (f", shortfall by class {shortfall}" if shortfall else "")


def import_csv(ctx: AppContext, profile: str, path: str) -> str:
    try:
        return f"Imported {import_human_labels(ctx.store, profile, Path(path))} human labels"
    except Exception as e:
        return f"Error: {e}"


def run_manual_test_ui(ctx: AppContext, profile: str, pairs_path: str, threshold: int | None) -> dict:
    p = ctx.profile(profile)
    universe = load_universe(p, Path(pairs_path))
    labeler = ctx.labeler(p, ctx.store.register_profile_version(p))
    return run_manual_test(ctx.store, labeler, universe, profile, threshold, ctx.settings.vlm_concurrency)


def build(ctx: AppContext) -> None:
    names = ctx.profile_names()
    default_profile = "sofas" if "sofas" in names else (names[0] if names else None)
    with gr.Row():
        queue = gr.Dropdown(ctx.store.queues(default_profile) if default_profile else [], label="Queue", allow_custom_value=True)
        reload_queues = gr.Button("Reload queues")
        position = gr.Markdown()
    with gr.Row():
        with gr.Column():
            a_img = gr.Image(label="Listing 1", interactive=False, height=420)
            a_title, a_desc, a_params = gr.Markdown(), gr.Textbox(label="Description", lines=6, interactive=False), gr.JSON(label="Parameters")
        with gr.Column():
            b_img = gr.Image(label="Listing 2", interactive=False, height=420)
            b_title, b_desc, b_params = gr.Markdown(), gr.Textbox(label="Description", lines=6, interactive=False), gr.JSON(label="Parameters")
    comment = gr.Textbox(label="Comment")
    with gr.Row():
        buttons = [gr.Button(str(c), variant="primary") for c in range(1, 6)]
    state = gr.State(None)
    with gr.Accordion("Manual test", open=False):
        with gr.Row():
            mt_profile = gr.Dropdown(names, value=default_profile, label="Profile")
            mt_pairs = gr.Textbox(label="Pairs file", value="/data/train_test_with_images/combined_labeled_df_balanced_small.csv")
            mt_per_class = gr.Number(label="Pairs per legacy class", value=40, precision=0)
            mt_from_csv = gr.Textbox(label="Fixed set CSV (optional)")
        with gr.Row():
            prepare = gr.Button("Prepare")
            human_csv = gr.Textbox(label="Human labels CSV (a_id,b_id,class[,comment])")
            do_import = gr.Button("Import CSV")
        with gr.Row():
            mt_threshold = gr.Number(label="Positive threshold (empty = not set)", precision=0, value=None)
            run = gr.Button("Run labeler")
        mt_message = gr.Markdown()
        mt_metrics = gr.JSON(label="Labeler vs human")
    outputs = [state, position, a_img, a_title, a_desc, a_params, b_img, b_title, b_desc, b_params, comment]

    def show(card: ReviewCard | None):
        if card is None:
            return None, "Queue is done 🎉", None, "", "", {}, None, "", "", {}, ""
        return ((card.a_id, card.b_id), f"**{card.position}**", card.a_image, f"**{card.a_title}**", card.a_description,
                card.a_params, card.b_image, f"**{card.b_title}**", card.b_description, card.b_params, "")

    def on_submit(cls):
        def handler(q, pair, text):
            if not q or not pair:
                return show(next_card(ctx, q) if q else None)
            return show(submit(ctx, q, pair[0], pair[1], cls, text))
        return handler

    queue.change(lambda q: show(next_card(ctx, q) if q else None), queue, outputs)
    reload_queues.click(lambda: gr.update(choices=ctx.store.queues(default_profile)), None, queue)
    for cls, button in enumerate(buttons, start=1):
        button.click(on_submit(cls), [queue, state, comment], outputs)
    prepare.click(lambda p, path, n, f: prepare_manual_test(ctx, p, path, n, f or None),
                  [mt_profile, mt_pairs, mt_per_class, mt_from_csv], mt_message)
    do_import.click(lambda p, path: import_csv(ctx, p, path), [mt_profile, human_csv], mt_message)
    run.click(lambda p, path, t: run_manual_test_ui(ctx, p, path, None if t in (None, "") else int(t)),
              [mt_profile, mt_pairs, mt_threshold], mt_metrics)
