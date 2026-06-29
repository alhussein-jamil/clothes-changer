"""CLI entry point — launches Gradio UI."""

import logging

from outfit_studio.ml.gpu_memory import configure_pytorch_memory

configure_pytorch_memory()

import torch

from outfit_studio.config import get_settings
from outfit_studio.content_config import get_app_name, get_tagline
from outfit_studio.ml.gpu_memory import gpu_free_gb, gpu_total_gb
from outfit_studio.ml.onnx_runtime import ensure_nvidia_cuda_libs, resolve_onnx_device
from outfit_studio.ui.gradio_app import GradioApp
from outfit_studio.utils import log_banner, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    settings = get_settings()

    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    vram = f"{gpu_free_gb():.1f}/{gpu_total_gb():.1f} GB free" if cuda else "n/a"
    if cuda:
        ensure_nvidia_cuda_libs()
    onnx_dev = resolve_onnx_device() if cuda else "cpu"

    log_banner(
        get_app_name(),
        get_tagline() or "AI outfit inpainting",
        f"http://{settings.host}:{settings.port}",
        f"compute: {device_name} | VRAM: {vram} | ONNX: {onnx_dev}",
    )

    logger.info(
        "Bootstrapping runtime (log_level=%s, debug=%s)",
        logging.getLevelName(settings.resolved_log_level()),
        settings.debug,
    )
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("CUDA optimizations enabled (cudnn.benchmark, tf32)")
    else:
        logger.warning("CUDA not available — running on CPU (slow)")

    settings.ensure_dirs()
    logger.info("Launching Gradio UI …")
    GradioApp().launch()


if __name__ == "__main__":
    main()
