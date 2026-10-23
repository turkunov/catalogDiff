import gradio as gr

from pseudolabel.app.main import build_app
from pseudolabel.app.tab_profiles import load_profile_text, preview_prompts, save_profile


def test_save_profile_rejects_invalid(app_ctx):
    before = load_profile_text(app_ctx, "sofas")
    ok, msg = save_profile(app_ctx, "sofas", "name: [unclosed")
    assert not ok and msg
    assert app_ctx.store.profile_versions("sofas") == []
    assert load_profile_text(app_ctx, "sofas") == before


def test_save_profile_rejects_missing_class(app_ctx):
    text = load_profile_text(app_ctx, "sofas").replace("\n  5:", "\n  6:")
    ok, msg = save_profile(app_ctx, "sofas", text)
    assert not ok and "class_definitions" in msg


def test_save_profile_creates_version(app_ctx):
    text = load_profile_text(app_ctx, "sofas").replace("category_label: Диваны", "category_label: Диваны (edited)", 1)
    ok, msg = save_profile(app_ctx, "sofas", text)
    assert ok and "version 1" in msg, msg
    assert "(edited)" in load_profile_text(app_ctx, "sofas")


def test_preview_prompts(app_ctx):
    visual, text, match = preview_prompts(app_ctx, "sofas")
    assert "upholstery" in visual and "Parameters to extract" in text and "диван роденго" in match


def test_build_app_has_five_english_tabs(app_ctx):
    demo = build_app(app_ctx)
    labels = [b.label for b in demo.blocks.values() if isinstance(b, gr.Tab)]
    assert labels == ["Profiles", "Pair labeling", "Iterations", "Manual review", "Labels"]
