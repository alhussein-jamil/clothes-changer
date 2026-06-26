"""Save per-step pipeline artifacts for debugging (images + run metadata)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from clothes_changer.config import Settings, get_settings
from clothes_changer.utils.image import mask_overlay

logger = logging.getLogger(__name__)


class PipelineDebugSession:
    """Write images and JSON metadata for one ``generate()`` run."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata: dict[str, Any] = {"steps": []}

    @classmethod
    def create(cls, settings: Settings, username: str) -> PipelineDebugSession | None:
        if not settings.pipeline_debug:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = settings.resolved_pipeline_debug_dir / f"{username}_{ts}"
        session = cls(run_dir)
        logger.info("Pipeline debug dumps → %s", session.root)
        return session

    def record(self, step: str, **fields: Any) -> None:
        self.metadata["steps"].append({"step": step, **fields})

    def save_meta(self) -> None:
        path = self.root / "run_metadata.json"
        path.write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")

    def save_image(self, rel_path: str, image: Image.Image) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    def save_mask(self, rel_path: str, mask: np.ndarray) -> None:
        arr = (mask > 0).astype(np.uint8) * 255
        self.save_image(rel_path, Image.fromarray(arr, mode="L"))

    def save_overlay(
        self,
        rel_path: str,
        image: Image.Image,
        person: np.ndarray,
        clothes: np.ndarray,
    ) -> None:
        self.save_image(rel_path, mask_overlay(image.convert("RGB"), person, clothes))

    def person_dir(self, index: int) -> Path:
        path = self.root / f"person_{index:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path


def maybe_debug_session(username: str) -> PipelineDebugSession | None:
    return PipelineDebugSession.create(get_settings(), username)
