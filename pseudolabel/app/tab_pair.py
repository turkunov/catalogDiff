"""Pair labeling tab: run the three-step labeler on a profile example or on two custom listings."""
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import gradio as gr

from pseudolabel.app.context import AppContext
from pseudolabel.data import Item, item_from_example
from pseudolabel.labeler import PairResult
from pseudolabel.metrics import THRESHOLD_MISSING, require_threshold
from pseudolabel.segment import cut_out

SEGMENTATION = [("Original photo", "none"), ("SAM3 · object by the profile's text prompt", "sam3"),
                ("BiRefNet · salient object", "birefnet")]


@dataclass
class PairView:
    visual_a: dict
    visual_b: dict
    params_a: dict
    params_b: dict
    reasoning: str
    cls: int | None
    binary: str
    latency_s: float
    error: str | None
    image_a: str = ""  # the photos the labeler saw (cut-outs when segmentation is on)
    image_b: str = ""


def _view(r: PairResult, threshold: int | None, a: Item, b: Item) -> PairView:
    if threshold is None:
        binary = THRESHOLD_MISSING
    elif r.cls is None:
        binary = "n/a"
    else:
        binary = "positive" if r.cls >= require_threshold(threshold) else "negative"
    return PairView(r.visual_a, r.visual_b, r.params_a, r.params_b, r.reasoning, r.cls, binary, round(r.latency_s, 1), r.error,
                    a.image, b.image)


def _segmented(ctx: AppContext, item: Item, method: str, seg) -> Item:
    """The item with its photo replaced by the cut-out; a separate id so the original photo's cache is not reused."""
    path, _ = cut_out(seg, item.image, ctx.settings.work_dir / "segment" / method)
    return replace(item, id=f"{item.id}@{method}", image=path)


def _label(ctx: AppContext, profile_name: str, a: dict, b: dict, threshold: int | None, segmentation: str = "none") -> PairView:
    if threshold is not None:
        require_threshold(threshold)
    profile = ctx.profile(profile_name)
    version = ctx.store.register_profile_version(profile)
    item_a, item_b = item_from_example("a", a, ctx.repo_root), item_from_example("b", b, ctx.repo_root)
    if segmentation and segmentation != "none":
        seg = ctx.segmenter_factory(segmentation, profile.object_prompt or profile.category_label)
        item_a, item_b = _segmented(ctx, item_a, segmentation, seg), _segmented(ctx, item_b, segmentation, seg)
    return _view(ctx.labeler(profile, version).label_pair(item_a, item_b), threshold, item_a, item_b)


def label_example(ctx: AppContext, profile_name: str, example_name: str, threshold: int | None,
                  segmentation: str = "none") -> PairView:
    profile = ctx.profile(profile_name)
    ex = next((e for e in profile.examples if e.name == example_name), None)
    if ex is None:
        raise ValueError(f"profile {profile_name!r} has no example {example_name!r}")
    key = f"example:{profile_name}:{ex.name}"
    return _label(ctx, profile_name, {**ex.a, "id": f"{key}:a"}, {**ex.b, "id": f"{key}:b"}, threshold, segmentation)


def _custom_id(d: dict) -> str:
    h = hashlib.sha1()
    image = str(d.get("image") or "")
    h.update(Path(image).read_bytes() if image and Path(image).is_file() else image.encode())
    h.update(json.dumps({k: d.get(k) for k in ("title", "description", "key_params")}, ensure_ascii=False, sort_keys=True).encode())
    return "custom:" + h.hexdigest()[:16]


def label_custom(ctx: AppContext, profile_name: str, a: dict, b: dict, threshold: int | None,
                 segmentation: str = "none") -> PairView:
    return _label(ctx, profile_name, {**a, "id": _custom_id(a)}, {**b, "id": _custom_id(b)}, threshold, segmentation)


def _threshold(value) -> int | None:
    """An empty field may arrive as 0 from Gradio; classes start at 1, so 0 means "not set"."""
    return None if value in (None, "", 0) else int(value)


def _outputs(v: PairView):
    cls = "—" if v.cls is None else str(v.cls)
    summary = f"### Class: {cls} · {v.binary} · {v.latency_s}s" + (f"\n\n**Error:** `{v.error}`" if v.error else "")
    return summary, v.image_a or None, v.image_b or None, v.reasoning, v.visual_a, v.visual_b, v.params_a, v.params_b


def build(ctx: AppContext) -> None:
    names = ctx.profile_names()
    with gr.Row():
        first = "sofas" if "sofas" in names else (names[0] if names else None)
        profile = gr.Dropdown(names, value=first, label="Profile")
        threshold = gr.Number(label="Positive threshold (empty = not set)", precision=0, value=None)
    segmentation = gr.Radio(SEGMENTATION, value="none", label="Object segmentation before labeling")
    initial = [e.name for e in ctx.profile(first).examples] if first else []
    with gr.Row():
        example = gr.Dropdown(initial, value=initial[0] if initial else None, label="Example")
        run_example = gr.Button("Label example", variant="primary")
    with gr.Accordion("Custom pair", open=False):
        with gr.Row():
            with gr.Column():
                a_img = gr.Image(type="filepath", label="Listing 1 photo")
                a_title = gr.Textbox(label="Title")
                a_desc = gr.Textbox(label="Description", lines=3)
                a_params = gr.Textbox(label="Key params (JSON)", value="{}")
            with gr.Column():
                b_img = gr.Image(type="filepath", label="Listing 2 photo")
                b_title = gr.Textbox(label="Title")
                b_desc = gr.Textbox(label="Description", lines=3)
                b_params = gr.Textbox(label="Key params (JSON)", value="{}")
        run_custom = gr.Button("Label custom pair", variant="primary")
    summary = gr.Markdown()
    with gr.Row():
        seen_a = gr.Image(label="Photo sent to the labeler · 1", interactive=False, height=300)
        seen_b = gr.Image(label="Photo sent to the labeler · 2", interactive=False, height=300)
    reasoning = gr.Textbox(label="Reasoning", lines=4)
    with gr.Row():
        visual_a, visual_b = gr.JSON(label="Photo attributes · 1"), gr.JSON(label="Photo attributes · 2")
    with gr.Row():
        params_a, params_b = gr.JSON(label="Parameters · 1"), gr.JSON(label="Parameters · 2")
    outputs = [summary, seen_a, seen_b, reasoning, visual_a, visual_b, params_a, params_b]
    failed = lambda e: (f"**Error:** `{e}`", None, None, "", {}, {}, {}, {})  # noqa: E731

    def examples_for(name):
        ex = [e.name for e in ctx.profile(name).examples] if name else []
        return gr.update(choices=ex, value=ex[0] if ex else None)

    def on_example(name, ex_name, t, seg):
        try:
            return _outputs(label_example(ctx, name, ex_name, _threshold(t), seg))
        except Exception as e:
            return failed(e)

    def on_custom(name, t, seg, ai, at, ad, ap, bi, bt, bd, bp):
        try:
            a = {"title": at, "description": ad, "key_params": json.loads(ap or "{}"), "image": ai}
            b = {"title": bt, "description": bd, "key_params": json.loads(bp or "{}"), "image": bi}
            return _outputs(label_custom(ctx, name, a, b, _threshold(t), seg))
        except Exception as e:
            return failed(e)

    profile.change(examples_for, profile, example)
    run_example.click(on_example, [profile, example, threshold, segmentation], outputs)
    run_custom.click(on_custom, [profile, threshold, segmentation, a_img, a_title, a_desc, a_params, b_img, b_title, b_desc,
                                 b_params], outputs)
