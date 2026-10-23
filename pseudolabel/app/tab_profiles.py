"""Profiles tab: edit a category profile as YAML, validate, save as a new version, preview prompts."""
import gradio as gr
import pandas as pd

from pseudolabel.app.context import AppContext
from pseudolabel.data import Item, item_from_example
from pseudolabel.profiles import Profile, ProfileError, parse_profile
from pseudolabel.prompts import match_prompt, text_params_prompt, visual_prompt


def load_profile_text(ctx: AppContext, name: str) -> str:
    return (ctx.settings.profiles_dir / f"{name}.yaml").read_text(encoding="utf-8")


def save_profile(ctx: AppContext, name: str, yaml_text: str) -> tuple[bool, str]:
    try:
        profile = parse_profile(yaml_text)
    except ProfileError as e:
        return False, str(e)
    if profile.name != name:
        return False, f"name in YAML must be '{name}', got '{profile.name}'"
    (ctx.settings.profiles_dir / f"{name}.yaml").write_text(yaml_text, encoding="utf-8")
    return True, f"Saved as version {ctx.store.register_profile_version(profile)}"


def versions_table(ctx: AppContext, name: str) -> pd.DataFrame:
    return pd.DataFrame(ctx.store.profile_versions(name), columns=["version", "created_at"])


def preview_prompts(ctx: AppContext, name: str) -> tuple[str, str, str]:
    return _preview(ctx, ctx.profile(name))


def preview_prompts_text(ctx: AppContext, yaml_text: str) -> tuple[str, str, str]:
    """Prompts of the profile currently in the editor, before saving."""
    return _preview(ctx, parse_profile(yaml_text))


def _preview(ctx: AppContext, p: Profile) -> tuple[str, str, str]:
    if p.examples:
        ex = p.examples[0]
        a, b = item_from_example("a", ex.a, ctx.repo_root), item_from_example("b", ex.b, ctx.repo_root)
    else:
        a = Item("a", "a", "<title of listing 1>", "<description of listing 1>", {}, "")
        b = Item("b", "b", "<title of listing 2>", "<description of listing 2>", {}, "")
    keys = [k.key for k in p.key_params]
    return (visual_prompt(p), text_params_prompt(p, a, keys),
            match_prompt(p, a, b, a.key_params, b.key_params, {"<attribute>": "<value>"}, {"<attribute>": "<value>"}))


def build(ctx: AppContext) -> None:
    names = ctx.profile_names()
    with gr.Row():
        name = gr.Dropdown(names, value=names[0] if names else None, label="Profile")
        status = gr.Markdown()
    editor = gr.Code(language="yaml", label="Profile YAML", lines=30,
                     value=load_profile_text(ctx, names[0]) if names else "")
    with gr.Row():
        save = gr.Button("Validate & save", variant="primary")
        preview = gr.Button("Preview prompts")
    versions = gr.Dataframe(versions_table(ctx, names[0]) if names else None, label="Saved versions", interactive=False)
    with gr.Accordion("Prompts", open=False):
        p_visual = gr.Textbox(label="Step 1 · visual attributes (photo only)", lines=10)
        p_text = gr.Textbox(label="Step 2 · parameters from text (text only)", lines=10)
        p_match = gr.Textbox(label="Step 3 · match", lines=20)

    def on_save(n, text):
        ok, msg = save_profile(ctx, n, text)
        return ("✅ " if ok else "❌ ") + msg, versions_table(ctx, n)

    def on_preview(text):
        try:
            return (*preview_prompts_text(ctx, text), "")
        except ProfileError as e:
            return "", "", "", f"❌ {e}"

    name.change(lambda n: (load_profile_text(ctx, n), versions_table(ctx, n)), name, [editor, versions])
    save.click(on_save, [name, editor], [status, versions])
    preview.click(on_preview, editor, [p_visual, p_text, p_match, status])
