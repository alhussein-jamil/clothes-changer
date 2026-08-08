"""Gradio UI layout constants and Game Studio–inspired theme."""

from __future__ import annotations

from typing import Final

import gradio as gr


class MaskEditor:
    ALPHA_VISIBLE_MIN: Final[int] = 8
    CHANNEL_MIN: Final[int] = 20
    COMPOSITE_DIFF_MIN: Final[int] = 6
    COMPOSITE_CHANNEL_BIAS_MIN: Final[int] = 8
    # Close stipple gaps when recovering clothes from airbrush composite only.
    COMPOSITE_CLOTHES_CLOSE_PX: Final[int] = 5
    FINGERPRINT_SIZE: Final[tuple[int, int]] = (64, 64)


class UI:
    EDITOR_CANVAS_SIZE: Final[tuple[int, int]] = (1000, 1000)
    PERSON_COLOR: Final[tuple[int, int, int, int]] = (255, 0, 0, 90)
    CLOTHES_COLOR: Final[tuple[int, int, int, int]] = (0, 255, 0, 110)
    BRUSH_DEFAULT_SIZE: Final[int] = 28
    MAX_EXAMPLES: Final[int] = 12
    HISTORY_GALLERY_LIMIT: Final[int] = 48
    HISTORY_CAPTION_MAX_LEN: Final[int] = 80
    HISTORY_GALLERY_HEIGHT: Final[int] = 420
    HISTORY_GALLERY_COLUMNS: Final[int] = 4
    CSS_MAX_WIDTH_PX: Final[int] = 1280
    STEPS_SLIDER_MIN: Final[int] = 10
    STEPS_SLIDER_MAX: Final[int] = 100
    CFG_SLIDER_MIN: Final[float] = 1.0
    CFG_SLIDER_MAX: Final[float] = 20.0
    CFG_SLIDER_STEP: Final[float] = 0.5
    DEFAULT_ADMIN_CREDITS_INPUT: Final[int] = 10
    LOGO_MAX_WIDTH_PX: Final[int] = 760
    LOGO_MAX_HEIGHT_PX: Final[int] = 120
    MARK_SIZE_PX: Final[int] = 40
    PROMPT_LINES: Final[int] = 3
    NEGATIVE_PROMPT_LINES: Final[int] = 2
    LOG_PREVIEW_LEN: Final[int] = 120
    DESCRIBE_DICT_KEYS_PREVIEW: Final[int] = 6


# Game Studio palette — dark olive chrome + warm terracotta accent.
_FONT_HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;700'
    '&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">'
)


def studio_head_html() -> str:
    """Font + color-scheme head fragment for ``gr.Blocks(head=…)``."""
    return _FONT_HEAD + '<meta name="color-scheme" content="dark">'


def build_gradio_theme() -> gr.themes.Base:
    """Gradio theme tokens aligned with the studio CSS variables."""
    return gr.themes.Base(
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
        primary_hue=gr.themes.Color(
            c50="#fff7ed",
            c100="#ffedd5",
            c200="#fed7aa",
            c300="#ffb183",
            c400="#f28c52",
            c500="#e8753a",
            c600="#d45f28",
            c700="#b34a1c",
            c800="#8a3816",
            c900="#422719",
            c950="#211009",
        ),
        secondary_hue="zinc",
        neutral_hue=gr.themes.Color(
            c50="#f4f2ec",
            c100="#e4e2da",
            c200="#b7b8b1",
            c300="#898d86",
            c400="#6b6f68",
            c500="#515851",
            c600="#343934",
            c700="#292d28",
            c800="#20231f",
            c900="#181b18",
            c950="#0d0f0d",
        ),
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_background_fill="#0d0f0d",
        body_background_fill_dark="#0d0f0d",
        body_text_color="#f4f2ec",
        body_text_color_dark="#f4f2ec",
        body_text_color_subdued="#898d86",
        body_text_color_subdued_dark="#898d86",
        background_fill_primary="#181b18",
        background_fill_primary_dark="#181b18",
        background_fill_secondary="#20231f",
        background_fill_secondary_dark="#20231f",
        border_color_primary="#343934",
        border_color_primary_dark="#343934",
        block_background_fill="#181b18",
        block_background_fill_dark="#181b18",
        block_border_color="#343934",
        block_border_color_dark="#343934",
        block_label_text_color="#898d86",
        block_label_text_color_dark="#898d86",
        block_title_text_color="#f4f2ec",
        block_title_text_color_dark="#f4f2ec",
        input_background_fill="#20231f",
        input_background_fill_dark="#20231f",
        input_border_color="#343934",
        input_border_color_dark="#343934",
        button_primary_background_fill="#f28c52",
        button_primary_background_fill_dark="#f28c52",
        button_primary_background_fill_hover="#ffb183",
        button_primary_background_fill_hover_dark="#ffb183",
        button_primary_text_color="#211009",
        button_primary_text_color_dark="#211009",
        button_secondary_background_fill="#20231f",
        button_secondary_background_fill_dark="#20231f",
        button_secondary_background_fill_hover="#292d28",
        button_secondary_background_fill_hover_dark="#292d28",
        button_secondary_text_color="#f4f2ec",
        button_secondary_text_color_dark="#f4f2ec",
        button_cancel_background_fill="#2a1616",
        button_cancel_background_fill_dark="#2a1616",
        button_cancel_text_color="#ff7e79",
        button_cancel_text_color_dark="#ff7e79",
        checkbox_label_background_fill="#20231f",
        checkbox_label_background_fill_dark="#20231f",
        checkbox_background_color="#f28c52",
        checkbox_background_color_dark="#f28c52",
        slider_color="#f28c52",
        slider_color_dark="#f28c52",
        color_accent="#f28c52",
        color_accent_soft="#422719",
    )


CUSTOM_CSS = f"""
:root {{
  color-scheme: dark;
  --os-bg: #0d0f0d;
  --os-chrome: #111311;
  --os-panel: #181b18;
  --os-surface: #20231f;
  --os-raised: #292d28;
  --os-line: #343934;
  --os-line-strong: #515851;
  --os-text: #f4f2ec;
  --os-muted: #898d86;
  --os-soft: #b7b8b1;
  --os-accent: #f28c52;
  --os-accent-strong: #ffb183;
  --os-accent-soft: #422719;
  --os-on-accent: #211009;
  --os-danger: #ff7e79;
  --os-success: #53c99a;
  --os-radius: 12px;
  --os-radius-sm: 8px;
  --os-font: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  --os-mono: "IBM Plex Mono", ui-monospace, monospace;
}}

html, body, .gradio-container {{
  font-family: var(--os-font) !important;
  background: var(--os-bg) !important;
  color: var(--os-text) !important;
}}

body {{
  background:
    radial-gradient(1200px 520px at 12% -10%, rgba(242, 140, 82, 0.10), transparent 55%),
    radial-gradient(900px 480px at 88% 0%, rgba(122, 168, 200, 0.06), transparent 50%),
    linear-gradient(180deg, #20231f 0%, var(--os-bg) 42%) !important;
  min-height: 100vh;
}}

.gradio-container {{
  max-width: {UI.CSS_MAX_WIDTH_PX}px !important;
  margin: 0 auto !important;
  padding-top: 0 !important;
}}

/* ── Header / brand ─────────────────────────────────────────────── */
#app-header {{
  display: block !important;
  width: 100% !important;
  margin: 0 0 18px !important;
  padding: 0 !important;
  animation: os-fade-in 420ms ease-out;
}}
#app-header .app-header {{
  border: 1px solid var(--os-line);
  border-radius: var(--os-radius);
  background: linear-gradient(180deg, #1c1f1b 0%, var(--os-panel) 100%);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
  overflow: hidden;
}}
#app-header .app-topbar {{
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 16px !important;
  padding: 14px 18px !important;
  border-bottom: 1px solid var(--os-line);
  background: var(--os-chrome);
}}
#app-header .app-brand {{
  display: flex !important;
  align-items: center !important;
  gap: 12px !important;
  min-width: 0;
}}
#app-header .app-mark {{
  width: {UI.MARK_SIZE_PX}px !important;
  height: {UI.MARK_SIZE_PX}px !important;
  border-radius: 5px !important;
  object-fit: contain !important;
  background: var(--os-surface);
  border: 1px solid var(--os-line);
  flex-shrink: 0;
}}
#app-header .app-brand-name {{
  margin: 0 !important;
  font-size: clamp(1.15rem, 2.2vw, 1.55rem) !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em !important;
  line-height: 1.15 !important;
  color: var(--os-text) !important;
  white-space: nowrap;
}}
#app-header .app-brand-name b,
#app-header .app-brand-name strong {{
  color: #898d86 !important;
  font-weight: 500 !important;
}}
#app-header .app-eyebrow {{
  margin: 0 !important;
  font-size: 10px !important;
  font-weight: 700 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--os-accent) !important;
  font-family: var(--os-mono) !important;
}}
#app-header .app-header-body {{
  padding: 16px 18px 18px !important;
}}
#app-header .app-header-title h1 {{
  display: none !important;
}}
#app-header .app-header-tagline {{
  margin: 0 !important;
  font-size: 0.95rem !important;
  line-height: 1.5 !important;
  color: var(--os-muted) !important;
  max-width: 52rem;
}}
#app-header .app-header-logo-wrap {{
  display: none !important;
}}
#app-header img.app-header-logo-legacy {{
  display: none !important;
}}

/* ── Panels / blocks ────────────────────────────────────────────── */
.block, .form, .panel-wrap {{
  border-radius: var(--os-radius) !important;
}}
.gr-group, .gr-box, .gr-panel, .svelte-1f354aw {{
  border-color: var(--os-line) !important;
}}
label, .block-label, .label-wrap span {{
  color: var(--os-muted) !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
}}
.prose, p, span, .markdown {{
  color: var(--os-soft);
}}

/* Tabs like Game Studio mode switch */
.tabs {{
  border: none !important;
}}
.tab-nav {{
  display: flex !important;
  gap: 4px !important;
  padding: 4px !important;
  margin-bottom: 14px !important;
  border: 1px solid var(--os-line) !important;
  border-radius: var(--os-radius-sm) !important;
  background: var(--os-raised) !important;
}}
.tab-nav button {{
  flex: 1 !important;
  border: 1px solid transparent !important;
  border-radius: 5px !important;
  background: transparent !important;
  color: var(--os-muted) !important;
  font-weight: 650 !important;
  font-size: 13px !important;
  padding: 10px 12px !important;
  transition: color 160ms ease, background 160ms ease, border-color 160ms ease;
}}
.tab-nav button:hover {{
  color: var(--os-text) !important;
}}
.tab-nav button.selected {{
  color: var(--os-text) !important;
  background: var(--os-surface) !important;
  border-color: var(--os-line-strong) !important;
  box-shadow: inset 0 -2px 0 var(--os-accent);
}}

/* Inputs */
input, textarea, select,
.gr-input, .gr-text-input textarea, .gr-text-input input {{
  background: var(--os-surface) !important;
  border: 1px solid var(--os-line) !important;
  border-radius: var(--os-radius-sm) !important;
  color: var(--os-text) !important;
}}
input:focus, textarea:focus, select:focus {{
  border-color: var(--os-accent) !important;
  box-shadow: 0 0 0 2px var(--os-accent-soft) !important;
}}

/* Buttons */
button.primary, .gr-button-primary {{
  background: var(--os-accent) !important;
  color: var(--os-on-accent) !important;
  border: 1px solid transparent !important;
  font-weight: 700 !important;
  border-radius: var(--os-radius-sm) !important;
  transition: background 160ms ease, transform 120ms ease;
}}
button.primary:hover, .gr-button-primary:hover {{
  background: var(--os-accent-strong) !important;
  transform: translateY(-1px);
}}
button.secondary, .gr-button-secondary {{
  background: var(--os-surface) !important;
  color: var(--os-text) !important;
  border: 1px solid var(--os-line) !important;
  border-radius: var(--os-radius-sm) !important;
}}
button.secondary:hover, .gr-button-secondary:hover {{
  border-color: var(--os-accent) !important;
  color: var(--os-accent-strong) !important;
  background: var(--os-accent-soft) !important;
}}
button.stop, .gr-button-stop {{
  background: #2a1616 !important;
  color: var(--os-danger) !important;
  border: 1px solid #6a3030 !important;
}}

#generate-btn {{
  letter-spacing: 0.02em;
}}
#studio-session-bar {{
  display: flex !important;
  gap: 10px !important;
  align-items: stretch !important;
  margin-bottom: 12px !important;
}}
#studio-session-bar textarea,
#studio-session-bar input {{
  font-family: var(--os-mono) !important;
  font-size: 12px !important;
  letter-spacing: 0.02em !important;
  text-transform: none !important;
}}

/* Image editor / slider stage */
#studio-generate-row .image-container,
.image-container, .image-frame, .upload-container {{
  border-radius: var(--os-radius-sm) !important;
  border: 1px solid var(--os-line) !important;
  background: linear-gradient(180deg, #20231f, #0b0d0b) !important;
}}
/* ImageEditor needs real height — transform animations + flex equal-height
   collapse the canvas to a bottom sliver (~few px). */
#studio-generate-row .image-container,
.image-container {{
  min-height: 420px !important;
  height: auto !important;
  overflow: visible !important;
}}
#studio-generate-row .image-container canvas,
#studio-generate-row .image-container img,
.image-container canvas,
.image-container img {{
  max-height: none !important;
}}
/* Soften Gradio ImageEditor dashed empty chrome */
.image-container .empty, .upload-container .wrap {{
  border-style: dashed !important;
  border-color: var(--os-line-strong) !important;
  color: var(--os-muted) !important;
}}
.toolbar-wrap, .image-container .toolbar {{
  background: rgba(17, 19, 17, 0.92) !important;
  border: 1px solid var(--os-line) !important;
  border-radius: var(--os-radius-sm) !important;
  backdrop-filter: blur(8px);
}}

/* Accordion */
.accordion {{
  border: 1px solid var(--os-line) !important;
  border-radius: var(--os-radius-sm) !important;
  background: var(--os-surface) !important;
}}
.accordion > .label-wrap {{
  color: var(--os-soft) !important;
}}

/* Gallery */
.gallery, .gallery-container {{
  border: 1px solid var(--os-line) !important;
  border-radius: var(--os-radius) !important;
  background: var(--os-panel) !important;
  padding: 8px !important;
}}

/* Examples chips */
.examples {{
  border: 1px solid var(--os-line) !important;
  border-radius: var(--os-radius-sm) !important;
  background: var(--os-panel) !important;
  padding: 10px !important;
}}

footer, .footer {{
  color: var(--os-muted) !important;
  font-size: 11px !important;
  opacity: 0.55;
}}
footer a {{
  color: var(--os-soft) !important;
}}

/* Hide noisy Gradio settings gear if present */
button.settings-button {{
  opacity: 0.45;
}}
button.settings-button:hover {{
  opacity: 1;
}}

@keyframes os-fade-in {{
  from {{ opacity: 0; transform: translateY(-6px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes os-rise {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

@media (max-width: 640px) {{
  #app-header .app-topbar {{
    flex-direction: column !important;
    align-items: flex-start !important;
  }}
  #app-header .app-brand-name {{
    white-space: normal;
  }}
}}
"""

EDITOR_CANVAS_SIZE = UI.EDITOR_CANVAS_SIZE
PERSON_COLOR = UI.PERSON_COLOR
CLOTHES_COLOR = UI.CLOTHES_COLOR
