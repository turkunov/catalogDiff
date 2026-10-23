"""Look of the app: dark surfaces, green accent, Outfit font, a hero with the project name.

The palette follows the team's AVILOOK demo page. The app is always dark: the same values are set for
Gradio's light and dark modes, and the page switches Gradio to its dark mode on load.
"""
import inspect

import gradio as gr

BG, SURFACE, SURFACE_2, SURFACE_3 = "#0e0e10", "#18181c", "#212127", "#2a2a32"
BORDER, TEXT, TEXT_DIM = "rgba(255,255,255,.06)", "#eaeaef", "#77778a"
ACCENT, ACCENT_LIGHT = "#16A34A", "#4ADE80"
ACCENT_LOW, ACCENT_MID, ACCENT_HIGH = "rgba(22,163,74,.10)", "rgba(22,163,74,.28)", "rgba(22,163,74,.50)"
GRADIENT = f"linear-gradient(135deg,{ACCENT_LIGHT},{ACCENT})"


_THEME_VARS = set(inspect.signature(gr.themes.Base.set).parameters)


def _both(**values: str) -> dict[str, str]:
    """The same value for the light and the dark variant of each theme variable (where Gradio has one)."""
    return {**values, **{f"{k}_dark": v for k, v in values.items() if f"{k}_dark" in _THEME_VARS}}


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Outfit"), "-apple-system", "BlinkMacSystemFont", "sans-serif"],
).set(
    block_border_width="1px",
    block_radius="14px",
    block_padding="16px",
    input_radius="8px",
    button_large_radius="10px",
    container_radius="14px",
    block_title_text_weight="600",
    **_both(
        body_background_fill=BG,
        body_text_color=TEXT,
        body_text_color_subdued=TEXT_DIM,
        background_fill_primary=SURFACE,
        background_fill_secondary=SURFACE_2,
        block_background_fill=SURFACE,
        block_border_color=BORDER,
        block_label_background_fill=SURFACE_2,
        block_label_text_color=TEXT_DIM,
        block_title_text_color=TEXT,
        panel_background_fill=SURFACE,
        border_color_primary=BORDER,
        border_color_accent=ACCENT,
        color_accent=ACCENT,
        color_accent_soft=ACCENT_LOW,
        input_background_fill=SURFACE_2,
        input_border_color=BORDER,
        input_border_color_focus=ACCENT,
        input_shadow_focus=f"0 0 0 1px {ACCENT}",
        button_primary_background_fill=GRADIENT,
        button_primary_background_fill_hover=GRADIENT,
        button_primary_text_color="#ffffff",
        button_primary_border_color="transparent",
        button_primary_shadow=f"0 5px 24px {ACCENT_MID}",
        button_primary_shadow_hover=f"0 7px 32px {ACCENT_HIGH}",
        button_secondary_background_fill=SURFACE_3,
        button_secondary_background_fill_hover=ACCENT_LOW,
        button_secondary_text_color=TEXT,
        button_secondary_border_color=BORDER,
        checkbox_background_color=SURFACE_3,
        checkbox_border_color="rgba(255,255,255,.28)",
        checkbox_background_color_selected=ACCENT,
        checkbox_border_color_selected=ACCENT,
        checkbox_label_background_fill=SURFACE_2,
        checkbox_label_background_fill_selected=ACCENT_LOW,
        checkbox_label_border_color=BORDER,
        checkbox_label_border_color_selected=ACCENT,
        checkbox_label_text_color_selected=TEXT,
        link_text_color=ACCENT_LIGHT,
        code_background_fill=SURFACE_2,
        table_even_background_fill=SURFACE,
        table_odd_background_fill=SURFACE_2,
        table_border_color=BORDER,
        loader_color=ACCENT,
        slider_color=ACCENT,
        accordion_text_color=TEXT,
        shadow_drop="none",
    ),
)

CSS = f"""
.gradio-container {{ max-width: 1360px !important; margin: 0 auto !important; padding: 28px 20px 80px !important; }}
footer {{ display: none !important; }}

.cd-hero {{ text-align: center; padding: 48px 24px; margin-bottom: 8px; position: relative; overflow: hidden;
  background: linear-gradient(145deg, {ACCENT_LOW}, transparent 70%); border: 1px solid {ACCENT_MID}; border-radius: 28px; }}
.cd-hero::before, .cd-hero::after {{ content: ''; position: absolute; border-radius: 50%; pointer-events: none; }}
.cd-hero::before {{ width: 400px; height: 400px; top: -200px; left: -80px;
  background: radial-gradient(circle, rgba(74,222,128,.12) 0%, transparent 70%); animation: cd-float 9s ease-in-out infinite; }}
.cd-hero::after {{ width: 300px; height: 300px; bottom: -150px; right: -40px;
  background: radial-gradient(circle, rgba(22,163,74,.08) 0%, transparent 70%); animation: cd-float 7s ease-in-out infinite reverse; }}
@keyframes cd-float {{ 0%, 100% {{ transform: translateY(0) rotate(0); }} 50% {{ transform: translateY(-16px) rotate(4deg); }} }}
.cd-hero-in {{ position: relative; z-index: 1; }}
.cd-hero-icon {{ display: inline-flex; align-items: center; justify-content: center; width: 68px; height: 68px;
  margin-bottom: 18px; border-radius: 20px; background: {GRADIENT};
  box-shadow: 0 10px 40px {ACCENT_HIGH}, 0 0 60px {ACCENT_LOW}; animation: cd-pulse 3s ease-in-out infinite; }}
@keyframes cd-pulse {{ 0%, 100% {{ transform: scale(1); opacity: 1; }} 50% {{ transform: scale(1.06); opacity: .85; }} }}
.cd-hero h1 {{ font-size: clamp(28px, 4.5vw, 46px); font-weight: 900; letter-spacing: -1px; margin: 0 0 10px;
  background: linear-gradient(135deg, #BBF7D0 0%, {ACCENT_LIGHT} 50%, {ACCENT} 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }}
.cd-hero p {{ color: {TEXT_DIM}; font-size: 15px; font-weight: 500; margin: 0 auto; }}

button[role="tab"] {{ font-weight: 600 !important; font-size: 14px !important; }}
button[role="tab"][aria-selected="true"] {{ color: {ACCENT_LIGHT} !important; }}
input[type="radio"]:not(:checked), input[type="checkbox"]:not(:checked) {{
  border: 1.5px solid rgba(255,255,255,.28) !important; }}
button.primary {{ font-weight: 700 !important; transition: transform .15s, box-shadow .15s !important; }}
button.primary:hover {{ transform: translateY(-1px); }}
button.primary:active {{ transform: scale(.98); }}
"""

# Gradio follows the browser's colour scheme; the palette above is dark, so dark mode is forced.
FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"

_ICON = (
    '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="3" y="3" width="12" height="12" rx="2.5"/><rect x="9" y="9" width="12" height="12" rx="2.5"/></svg>'
)


def hero(subtitle: str) -> str:
    return (f'<div class="cd-hero"><div class="cd-hero-in"><div class="cd-hero-icon">{_ICON}</div>'
            f"<h1>CatalogDiff</h1><p>{subtitle}</p></div></div>")
