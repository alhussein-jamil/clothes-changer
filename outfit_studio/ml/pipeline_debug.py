"""Save per-step pipeline artifacts for debugging (images + run metadata)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from outfit_studio.config import Settings
from outfit_studio.constants import MASK_ON
from outfit_studio.utils.image import mask_overlay

logger = logging.getLogger(__name__)


class PipelineDebugSession:
    """Write images and JSON metadata for one debug run (segmentation + generation)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata: dict[str, Any] = {"events": []}

    @classmethod
    def open_or_create(
        cls,
        settings: Settings,
        username: str,
        existing_dir: str | Path | None = None,
    ) -> tuple[PipelineDebugSession | None, str | None]:
        """Reuse an active run folder or create ``{username}_{timestamp}``."""
        if not settings.pipeline_debug:
            return None, None if existing_dir is None else str(existing_dir)

        if existing_dir:
            root = Path(existing_dir)
            if root.is_dir():
                logger.debug("Reusing pipeline debug folder %s", root)
                return cls(root), str(root.resolve())

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = settings.resolved_pipeline_debug_dir / f"{username}_{ts}"
        session = cls(root)
        session.metadata["username"] = username
        logger.info("Pipeline debug dumps → %s", session.root)
        return session, str(root.resolve())

    @classmethod
    def create(cls, settings: Settings, username: str) -> PipelineDebugSession | None:
        session, _ = cls.open_or_create(settings, username, None)
        return session

    def subfolder(self, name: str) -> PipelineDebugSession:
        """Nested session, e.g. ``segmentation/`` or ``generation/`` under the run root."""
        child = PipelineDebugSession(self.root / name)
        child.metadata["parent"] = str(self.root)
        child.metadata["phase"] = name
        return child

    def record(self, step: str, **fields: Any) -> None:
        self.metadata["events"].append({"step": step, **fields})

    def save_meta(self) -> None:
        path = self.root / "run_metadata.json"
        path.write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")

    def save_image(self, rel_path: str, image: Image.Image) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    def save_mask(self, rel_path: str, mask: np.ndarray) -> None:
        arr = (mask > 0).astype(np.uint8) * MASK_ON
        self.save_image(rel_path, Image.fromarray(arr, mode="L"))

    def save_tensor_mask(self, rel_path: str, mask: torch.Tensor) -> None:
        arr = mask.detach().cpu().numpy()
        if arr.ndim > 2:
            arr = arr.squeeze()
        self.save_mask(rel_path, (arr > 0).astype(np.uint8))

    def save_overlay(
        self,
        rel_path: str,
        image: Image.Image,
        person: np.ndarray,
        clothes: np.ndarray,
    ) -> None:
        self.save_image(rel_path, mask_overlay(image.convert("RGB"), person, clothes))
