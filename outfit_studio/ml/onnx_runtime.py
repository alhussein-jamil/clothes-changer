"""ONNX Runtime device selection for rtmlib pose/detection models."""

from __future__ import annotations

import contextlib
import logging
import os
import site
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_NVIDIA_LIB_SUBDIRS = (
    "cublas",
    "cuda_runtime",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvrtc",
)


def ensure_nvidia_cuda_libs() -> None:
    """Expose pip-installed NVIDIA CUDA 12 libraries to onnxruntime-gpu.

    PyTorch cu130 wheels bundle CUDA 13, while PyPI ``onnxruntime-gpu`` is built
    for CUDA 12. The companion ``nvidia-*-cu12`` packages supply the missing
    ``libcublasLt.so.12``, ``libcufft.so.11``, etc. when they are on
    ``LD_LIBRARY_PATH``. See ``NVIDIA_CUDA12_LIBS`` in the Makefile.
    """
    dirs: list[str] = []
    for base in site.getsitepackages():
        nvidia_root = Path(base) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for name in _NVIDIA_LIB_SUBDIRS:
            lib_dir = nvidia_root / name / "lib"
            if lib_dir.is_dir():
                dirs.append(str(lib_dir))

    if not dirs:
        logger.debug("No pip NVIDIA CUDA lib dirs found for ONNX Runtime")
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    missing = [d for d in dirs if d not in existing.split(":")]
    if not missing:
        logger.debug("NVIDIA CUDA libs already on LD_LIBRARY_PATH (%d dirs)", len(dirs))
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + ([existing] if existing else []))
    logger.info(
        "Prepended %d NVIDIA CUDA lib dirs to LD_LIBRARY_PATH for ONNX Runtime",
        len(missing),
    )


@lru_cache
def resolve_onnx_device() -> str:
    """Return ``cuda`` only when the ONNX Runtime CUDA EP can actually load."""
    if not torch.cuda.is_available():
        logger.debug("ONNX device: cpu (CUDA unavailable)")
        return "cpu"

    ensure_nvidia_cuda_libs()

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; pose/detection will use CPU")
        return "cpu"

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        logger.info("ONNX Runtime has no CUDA provider; pose/detection will use CPU")
        return "cpu"

    cache_dir = Path.home() / ".cache/rtmlib/hub/checkpoints"
    candidates = sorted(cache_dir.glob("*.onnx"))
    if not candidates:
        logger.debug("No cached ONNX checkpoints yet; assuming CUDA for pose models")
        return "cuda"

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 4  # fatal only during probe
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        try:
            session = ort.InferenceSession(
                str(candidates[0]),
                sess_options=session_options,
                providers=["CUDAExecutionProvider"],
            )
            providers = session.get_providers()
            del session
            if "CUDAExecutionProvider" not in providers:
                logger.info(
                    "ONNX Runtime CUDA unavailable (using %s); "
                    "pose/detection will run on CPU. "
                    "Run `make install-fast` to install CUDA 12 libs for ORT.",
                    providers[0] if providers else "CPU",
                )
                return "cpu"
            logger.info("ONNX Runtime CUDA provider verified")
            return "cuda"
        except Exception:
            logger.info(
                "ONNX Runtime CUDA libraries unavailable; pose/detection will use CPU. "
                "Run `make install-fast` to install CUDA 12 libs for ORT."
            )
            return "cpu"
