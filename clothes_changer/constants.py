"""Named constants — single source for values that are not user settings."""

from __future__ import annotations

from typing import Final

# --- Binary masks ---
MASK_ON: Final[int] = 255
MASK_OFF: Final[int] = 0

# --- SegFormer label IDs (mattmdjaga/segformer_b2_clothes) ---
PERSON_SEGFORMER_CATEGORIES: Final[tuple[int, ...]] = (1, 2, 3, 11, 12, 13, 14, 15, 9, 10, 16)
CLOTHES_SEGFORMER_CATEGORIES: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 16, 17)

# --- U2NET cloth segmentation ---
U2NET_INPUT_SIZE: Final[int] = 768
U2NET_OUTPUT_CLASSES: Final[int] = 4
U2NET_INPUT_CHANNELS: Final[int] = 3
U2NET_NORMALIZE_MEAN: Final[float] = 0.5
U2NET_NORMALIZE_STD: Final[float] = 0.5
U2NET_TENSOR_CHANNELS: Final[tuple[int, ...]] = (1, 3, 18)

# --- Crop / mask morphology ---
CROP_BOX_PADDING_RATIO: Final[float] = 0.1
BLEND_MASK_GROW_DIVISOR: Final[int] = 30
BLEND_FEATHER_DIVISOR: Final[int] = 4
INSTANCE_MASK_GROW_DIVISOR: Final[int] = 60
DEFAULT_MASK_GROW_PX: Final[int] = 5

# --- Stable Diffusion ---
LATENT_ALIGN: Final[int] = 8
MIN_LATENT_SIDE: Final[int] = 64
CLIP_MAX_TOKENS: Final[int] = 77

# --- Random seeds ---
SEED_MAX: Final[int] = 999_999
DEFAULT_SEED: Final[int] = 42

# --- HTTP downloads ---
HTTP_DOWNLOAD_TIMEOUT_S: Final[int] = 60
HTTP_DOWNLOAD_CHUNK_BYTES: Final[int] = 8192
DOWNLOAD_SIZE_TOLERANCE: Final[float] = 0.99
HTTP_USER_AGENT: Final[str] = "clothes-changer/1.0"
BYTES_PER_MIB: Final[int] = 1_048_576

# --- VRAM budget estimates (GB) ---
VRAM_SEGMENTATION_PEAK_GB: Final[float] = 2.0
VRAM_INPAINT_SDXL_GB: Final[float] = 10.0
VRAM_INPAINT_CONTROLNET_GB: Final[float] = 6.0
VRAM_INPAINT_PLAIN_GB: Final[float] = 4.5
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
    PROMPT_LINES: Final[int] = 3
    NEGATIVE_PROMPT_LINES: Final[int] = 2
    LOG_PREVIEW_LEN: Final[int] = 120
    DESCRIBE_DICT_KEYS_PREVIEW: Final[int] = 6


CUSTOM_CSS = f"""
.gradio-container {{ max-width: {UI.CSS_MAX_WIDTH_PX}px !important; margin: auto; }}
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
MIN_PASSWORD_LENGTH: Final[int] = 5
DEFAULT_NEW_USER_CREDITS: Final[int] = 10
HISTORY_DB_LIMIT: Final[int] = 50
