"""Checkpoint naming and architecture helpers."""

from __future__ import annotations

from pathlib import Path

SDXL_NAME_HINTS = ("sdxl", "xl_inpaint", "xl-inpaint", "xl_inpainting")

# Checkpoints above ~3.5 GB are treated as SDXL when the filename is ambiguous.
_SDXL_SIZE_BYTES = 3_500_000_000


def is_sdxl_model_name(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in SDXL_NAME_HINTS)


def is_sdxl_checkpoint(name: str, path: Path) -> bool:
    if is_sdxl_model_name(name):
        return True
    return path.is_file() and path.stat().st_size > _SDXL_SIZE_BYTES
