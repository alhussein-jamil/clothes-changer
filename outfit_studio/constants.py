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
