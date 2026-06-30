"""Gradio UI layout constants and theme CSS."""

from __future__ import annotations

from typing import Final


class MaskEditor:
    ALPHA_VISIBLE_MIN: Final[int] = 8
    CHANNEL_MIN: Final[int] = 20
    COMPOSITE_DIFF_MIN: Final[int] = 6
    COMPOSITE_CHANNEL_BIAS_MIN: Final[int] = 8
    FINGERPRINT_SIZE: Final[tuple[int, int]] = (64, 64)


class UI:
    EDITOR_CANVAS_SIZE: Final[tuple[int, int]] = (1000, 1000)
    PERSON_COLOR: Final[tuple[int, int, int, int]] = (255, 0, 0, 100)
    CLOTHES_COLOR: Final[tuple[int, int, int, int]] = (0, 255, 0, 100)
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
    PROMPT_LINES: Final[int] = 3
    NEGATIVE_PROMPT_LINES: Final[int] = 2
    LOG_PREVIEW_LEN: Final[int] = 120
    DESCRIBE_DICT_KEYS_PREVIEW: Final[int] = 6


CUSTOM_CSS = f"""
.gradio-container {{ max-width: {UI.CSS_MAX_WIDTH_PX}px !important; margin: auto; }}
#app-header {{
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  width: 100% !important;
  padding: 16px 0 !important;
}}
#app-header .app-header-brand {{
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 18px !important;
  flex-wrap: wrap !important;
}}
#app-header .app-header-logo-wrap {{
  flex: 0 0 auto !important;
}}
#app-header .app-header-title {{
  flex: 0 1 auto !important;
  text-align: left !important;
}}
#app-header .app-header-title h1 {{
  margin: 0 !important;
  font-size: 1.75rem !important;
  font-weight: 700 !important;
  line-height: 1.2 !important;
  letter-spacing: -0.02em !important;
  color: #172033 !important;
}}
.dark #app-header .app-header-title h1 {{
  color: #e8eaed !important;
}}
#app-header .app-header-tagline {{
  margin: 6px 0 0 !important;
  font-size: 0.95rem !important;
  line-height: 1.4 !important;
  color: #5d6678 !important;
}}
.dark #app-header .app-header-tagline {{
  color: #9aa0a8 !important;
}}
#app-header img {{
  display: block !important;
  margin: 0 !important;
  max-width: min({UI.LOGO_MAX_WIDTH_PX}px, 100%);
  height: auto;
  max-height: {UI.LOGO_MAX_HEIGHT_PX}px;
}}
@media (max-width: 520px) {{
  #app-header .app-header-brand {{
    flex-direction: column !important;
  }}
  #app-header .app-header-title {{
    text-align: center !important;
  }}
}}
"""

EDITOR_CANVAS_SIZE = UI.EDITOR_CANVAS_SIZE
PERSON_COLOR = UI.PERSON_COLOR
CLOTHES_COLOR = UI.CLOTHES_COLOR
