"""Stable Diffusion ControlNet inpainting — matches original ClothLess model loading."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import requests
import torch
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
)
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from PIL import Image
from tqdm import tqdm

from clothes_changer.config import Settings, get_settings
from clothes_changer.content_config import get_checkpoint_urls, get_default_inpaint_model
from clothes_changer.ml.checkpoints import is_sdxl_checkpoint
from clothes_changer.ml.gpu_memory import free_cuda_cache
from clothes_changer.utils.logging import log_duration

logger = logging.getLogger(__name__)

MODEL_ALIASES: dict[str, list[str]] = {
    "realisticVisionV60B1_v51HyperInpaintVAE.safetensors": [
        "realisticVisionV60B1_v51HyperInpaintVAE_full.safetensors",
    ],
}

StepProgressCallback = Callable[[int, int], None]


class InpaintEngine:
    """Lazy-loaded SD inpainting with optional ControlNet (original defaults)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._pipe = None
        self._current_model: str | None = None
        self._use_controlnet = False
        self._architecture: str = "sd15"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        logger.info("InpaintEngine ready (device=%s, dtype=%s)", self.device, self.dtype)

    def _discover_local_models(self) -> list[str]:
        models_dir = self.settings.resolved_models_dir
        if not models_dir.is_dir():
            logger.debug("No models directory at %s", models_dir)
            return []
        found: list[str] = []
        for pattern in ("*.safetensors", "*.ckpt"):
            found.extend(p.name for p in sorted(models_dir.glob(pattern)))
        logger.debug("Discovered %d local checkpoint(s)", len(found))
        return found

    def list_models(self) -> list[dict]:
        models: list[dict] = []
        local = self._discover_local_models()
        all_names = list(local)
        for name in get_checkpoint_urls():
            if name not in all_names:
                all_names.append(name)

        for name in all_names:
            path = self._resolve_local_model(name)
            arch = "sdxl" if path.is_file() and is_sdxl_checkpoint(name, path) else "sd15"
            source = "local" if path.is_file() else "download"
            models.append(
                {
                    "id": name,
                    "name": Path(name).stem,
                    "source": source,
                    "arch": arch,
                }
            )

        if not models:
            models.append(
                {
                    "id": self.settings.inpaint_model,
                    "name": Path(self.settings.inpaint_model).stem,
                    "source": "download",
                    "arch": "sd15",
                }
            )
        return models

    def default_model_id(self) -> str:
        models = self.list_models()
        preferred = self.settings.inpaint_model
        ids = [m["id"] for m in models]
        if preferred in ids:
            return preferred
        fallback = get_default_inpaint_model()
        if fallback in ids:
            return fallback
        for m in models:
            if m["arch"] == "sd15":
                return m["id"]
        return models[0]["id"]

    def model_architecture(self, model_id: str) -> str:
        path = self._resolve_local_model(model_id)
        if path.is_file():
            return "sdxl" if is_sdxl_checkpoint(model_id, path) else "sd15"
        return "sd15"

    def download_model(self, model_path: Path) -> Path:
        if model_path.is_file():
            return model_path
        url = get_checkpoint_urls().get(model_path.name)
        if not url:
            msg = f"Model {model_path.name} not found locally and has no download URL"
            raise FileNotFoundError(msg)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s from %s", model_path.name, url)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            model_path.open("wb") as f,
            tqdm(
                desc=model_path.name,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(8192):
                size = f.write(chunk)
                bar.update(size)
        return model_path

    def _resolve_local_model(self, model_id: str) -> Path:
        primary = self.settings.resolved_models_dir / model_id
        if primary.is_file():
            return primary
        for alias in MODEL_ALIASES.get(model_id, []):
            alias_path = self.settings.resolved_models_dir / alias
            if alias_path.is_file():
                return alias_path
        return primary

    def _resolve_model_path(self, model_id: str) -> str:
        local = self._resolve_local_model(model_id)
        if local.is_file():
            return str(local)
        self.download_model(local)
        return str(local)

    def unload(self) -> None:
        self._unload()

    def _unload(self) -> None:
        if self._pipe is not None:
            logger.info("Unloading inpaint pipeline (model=%s)", self._current_model)
            del self._pipe
            self._pipe = None
            free_cuda_cache()

    def load(self, model_id: str | None = None, use_controlnet: bool | None = None) -> None:
        model_id = model_id or self.default_model_id()
        arch = self.model_architecture(model_id)
        use_controlnet = (
            use_controlnet if use_controlnet is not None else self.settings.use_controlnet
        )
        if arch == "sdxl" and use_controlnet:
            logger.info("Disabling ControlNet for SDXL checkpoint %s", model_id)
            use_controlnet = False

        if (
            self._pipe is not None
            and self._current_model == model_id
            and self._use_controlnet == use_controlnet
            and self._architecture == arch
        ):
            logger.debug("Reusing loaded inpaint pipeline (%s)", model_id)
            return

        self._unload()
        model_path = self._resolve_model_path(model_id)
        logger.info(
            "Loading inpaint model: %s arch=%s controlnet=%s",
            model_path,
            arch,
            use_controlnet,
        )

        with log_duration(logger, "load inpaint pipeline", model=model_id, arch=arch):
            disable_progress_bars()
            try:
                if arch == "sdxl":
                    if model_path.endswith(".safetensors"):
                        self._pipe = StableDiffusionXLInpaintPipeline.from_single_file(
                            model_path,
                            torch_dtype=self.dtype,
                        )
                    else:
                        self._pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                            model_path,
                            torch_dtype=self.dtype,
                        )
                elif use_controlnet and self.device.type == "cuda":
                    controlnet = ControlNetModel.from_pretrained(
                        self.settings.controlnet_model,
                        torch_dtype=self.dtype,
                    )
                    self._pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                        model_path,
                        controlnet=controlnet,
                        torch_dtype=self.dtype,
                        use_safetensors=model_path.endswith(".safetensors"),
                        safety_checker=None,
                    )
                elif model_path.endswith((".safetensors", ".ckpt")):
                    self._pipe = StableDiffusionInpaintPipeline.from_single_file(
                        model_path,
                        torch_dtype=self.dtype,
                        use_safetensors=model_path.endswith(".safetensors"),
                        safety_checker=None,
                    )
                else:
                    self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        model_path,
                        torch_dtype=self.dtype,
                        safety_checker=None,
                    )
            finally:
                enable_progress_bars()

            assert self._pipe is not None
            if hasattr(self._pipe, "safety_checker"):
                self._pipe.safety_checker = None
            if hasattr(self._pipe, "set_progress_bar_config"):
                self._pipe.set_progress_bar_config(disable=True)
            self._pipe.scheduler = DPMSolverMultistepScheduler(
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++",
            )
            self._pipe = self._pipe.to(self.device)
            self._enable_fast_attention(self._pipe)

        self._current_model = model_id
        self._use_controlnet = use_controlnet
        self._architecture = arch
        logger.info("Inpaint pipeline loaded on %s", self.device)

    def _truncate_prompts(self, prompt: str, negative_prompt: str) -> tuple[str, str]:
        """Keep prompts within CLIP's 77-token limit."""
        if self._pipe is None or not hasattr(self._pipe, "tokenizer"):
            return prompt, negative_prompt
        tokenizer = self._pipe.tokenizer
        max_len = getattr(tokenizer, "model_max_length", 77)

        def _truncate(text: str) -> str:
            ids = tokenizer.encode(text, truncation=True, max_length=max_len)
            return tokenizer.decode(ids, skip_special_tokens=True)

        return _truncate(prompt), _truncate(negative_prompt)

    @staticmethod
    def _enable_fast_attention(pipe) -> None:
        if not torch.cuda.is_available():
            return
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xFormers memory-efficient attention enabled")
            return
        except Exception:
            pass
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0

            pipe.unet.set_attn_processor(AttnProcessor2_0())
            if getattr(pipe, "controlnet", None) is not None:
                pipe.controlnet.set_attn_processor(AttnProcessor2_0())
            logger.info("PyTorch SDPA attention enabled")
        except Exception as e:
            logger.warning("Fast attention not available: %s", e)

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
        control_image: Image.Image | None = None,
        width: int | None = None,
        height: int | None = None,
        strength: float = 1.0,
        on_step: StepProgressCallback | None = None,
    ) -> Image.Image:
        if self._pipe is None:
            self.load()
        assert self._pipe is not None

        steps = steps or self.settings.inpaint_steps
        guidance_scale = guidance_scale or self.settings.guidance_scale

        prompt, negative_prompt = self._truncate_prompts(prompt, negative_prompt)

        orig_w, orig_h = image.size
        infer_w = width or max(64, (orig_w // 8) * 8)
        infer_h = height or max(64, (orig_h // 8) * 8)
        logger.info(
            "Inpaint %dx%d → %dx%d | steps=%d cfg=%.1f controlnet=%s",
            orig_w,
            orig_h,
            infer_w,
            infer_h,
            steps,
            guidance_scale,
            self._use_controlnet and control_image is not None,
        )

        kwargs: dict = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "mask_image": mask,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "width": infer_w,
            "height": infer_h,
            "strength": strength,
        }

        if self._use_controlnet and control_image is not None:
            kwargs["control_image"] = control_image

        if on_step is not None:
            total_steps = steps

            def _callback_on_step_end(_pipe, step: int, _timestep, callback_kwargs):  # noqa: ANN001
                on_step(step, total_steps)
                return callback_kwargs

            kwargs["callback_on_step_end"] = _callback_on_step_end
            kwargs["callback_on_step_end_tensor_inputs"] = []

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.device.type == "cuda",
            ),
        ):
            result = self._pipe(**kwargs).images[0]

        if result.size != (orig_w, orig_h):
            result = result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        return result


_engine: InpaintEngine | None = None


def get_inpaint_engine() -> InpaintEngine:
    global _engine
    if _engine is None:
        logger.debug("Creating InpaintEngine singleton")
        _engine = InpaintEngine()
    return _engine
