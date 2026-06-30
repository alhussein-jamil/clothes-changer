"""Named constants — single source for values that are not user settings."""

from __future__ import annotations

from typing import Final

# --- Binary masks ---
MASK_ON: Final[int] = 255
MASK_OFF: Final[int] = 0

# --- Human parser label IDs (fashn-ai/fashn-human-parser) ---
# 0 background | 1 face | 2 hair | 3 top | 4 dress | 5 skirt | 6 pants | 7 belt
# 8 bag | 9 hat | 10 scarf | 11 glasses | 12 arms | 13 hands | 14 legs | 15 feet
# 16 torso | 17 jewelry
PERSON_PARSER_CATEGORIES: Final[tuple[int, ...]] = tuple(range(1, 18))
CLOTHES_PARSER_CATEGORIES: Final[tuple[int, ...]] = (3, 4, 5, 6, 7, 8, 10)

# --- Segmentation post-processing ---
SEGMENTATION_CLOTHES_CONFIDENCE: Final[float] = 0.35
SEGMENTATION_MIN_COMPONENT_AREA: Final[int] = 64
# Ellipse kernel size for grow_mask; covers parser edge mislabels (torso/arms at hem).
SEGMENTATION_CLOTHES_EDGE_GROW_PX: Final[int] = 7

# --- Crop / mask morphology ---
CROP_BOX_PADDING_RATIO: Final[float] = 0.1
BLEND_MASK_GROW_DIVISOR: Final[int] = 30
BLEND_FEATHER_DIVISOR: Final[int] = 4
INSTANCE_MASK_GROW_DIVISOR: Final[int] = 60
MIN_INSTANCE_CLOTHES_PIXELS: Final[int] = 500
MIN_POSE_IMAGE_SIDE: Final[int] = 32
DEFAULT_MASK_GROW_PX: Final[int] = 5

# --- Stable Diffusion ---
LATENT_ALIGN: Final[int] = 8
MIN_LATENT_SIDE: Final[int] = 64
CLIP_MAX_TOKENS: Final[int] = 77
INPAINT_STRENGTH: Final[float] = 1.0

# --- Random seeds ---
SEED_MAX: Final[int] = 999_999
DEFAULT_SEED: Final[int] = 42

# --- HTTP downloads ---
HTTP_DOWNLOAD_TIMEOUT_S: Final[int] = 60
HTTP_DOWNLOAD_CHUNK_BYTES: Final[int] = 8192
DOWNLOAD_SIZE_TOLERANCE: Final[float] = 0.99
HTTP_USER_AGENT: Final[str] = "outfit-studio/1.0"
BYTES_PER_MIB: Final[int] = 1_048_576

# --- VRAM budget estimates (GB) ---
VRAM_SEGMENTATION_PEAK_GB: Final[float] = 3.5
VRAM_INPAINT_SDXL_GB: Final[float] = 10.0
VRAM_INPAINT_CONTROLNET_GB: Final[float] = 6.0
VRAM_INPAINT_PLAIN_GB: Final[float] = 4.5
VRAM_POSE_PEAK_GB: Final[float] = 0.5
BYTES_PER_GB: Final[int] = 1024**3
BYTES_PER_KIB: Final[int] = 1024


# --- Gradio ImageEditor mask parsing ---
class MaskEditor:
    ALPHA_VISIBLE_MIN: Final[int] = 8
    CHANNEL_MIN: Final[int] = 20
    COMPOSITE_DIFF_MIN: Final[int] = 6
    COMPOSITE_CHANNEL_BIAS_MIN: Final[int] = 8
    FINGERPRINT_SIZE: Final[tuple[int, int]] = (64, 64)


# --- UI layout & controls ---
class UI:
    EDITOR_CANVAS_SIZE: Final[tuple[int, int]] = (1000, 1000)
    PERSON_COLOR: Final[tuple[int, int, int, int]] = (255, 0, 0, 100)
    CLOTHES_COLOR: Final[tuple[int, int, int, int]] = (0, 255, 0, 100)
    MAX_EXAMPLES: Final[int] = 12
    HISTORY_GALLERY_LIMIT: Final[int] = 48
    HISTORY_CAPTION_MAX_LEN: Final[int] = 80
    HISTORY_POLL_INTERVAL_S: Final[int] = 10
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


# --- Generation pipeline progress (fractions 0–1) ---
class GenerateProgress:
    PREP_START: Final[float] = 0.05
    SEGMENT: Final[float] = 0.10
    DETECT_PEOPLE: Final[float] = 0.15
    PREP_END: Final[float] = 0.18
    PERSON_START: Final[float] = 0.20
    PERSON_SPAN: Final[float] = 0.70
    SAVE: Final[float] = 0.95


class PersonProgress:
    PREP: Final[float] = 0.0
    POSE_DETECT: Final[float] = 0.06
    POSE_GUIDE: Final[float] = 0.09
    PREP_AREA: Final[float] = 0.06
    LOAD_MODEL: Final[float] = 0.11
    DIFFUSION_START: Final[float] = 0.12
    DIFFUSION_SPAN: Final[float] = 0.76
    BLEND: Final[float] = 0.90


# --- Auth / database defaults ---
MIN_PASSWORD_LENGTH: Final[int] = 8
DEFAULT_NEW_USER_CREDITS: Final[int] = 10
DEFAULT_ADMIN_BOOTSTRAP_CREDITS: Final[int] = 100
HISTORY_DB_LIMIT: Final[int] = 50

# --- UI convenience aliases (avoid importing UI. everywhere) ---
EDITOR_CANVAS_SIZE = UI.EDITOR_CANVAS_SIZE
PERSON_COLOR = UI.PERSON_COLOR
CLOTHES_COLOR = UI.CLOTHES_COLOR
