"""Stable Diffusion ControlNet inpainting for clothing edits."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Literal

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

from outfit_studio.config import Settings, get_settings
from outfit_studio.constants import (
    BYTES_PER_KIB,
    BYTES_PER_MIB,
    CLIP_MAX_TOKENS,
    DOWNLOAD_SIZE_TOLERANCE,
    HTTP_DOWNLOAD_CHUNK_BYTES,
    HTTP_DOWNLOAD_TIMEOUT_S,
    HTTP_USER_AGENT,
    MASK_ON,
)
from outfit_studio.content_config import (
    get_checkpoint_urls,
    get_default_inpaint_model,
    get_model_aliases,
)
from outfit_studio.ml.checkpoints import (
    checkpoint_architecture,
    clear_checkpoint_cache,
    inpaint_checkpoint_listable,
    inpaint_checkpoint_valid,
    is_hub_model_id,
)
from outfit_studio.ml.compile_cache import load_artifacts, save_artifacts
from outfit_studio.ml.gpu_memory import free_cuda_cache, model_load_lock
from outfit_studio.ui.operation_control import OperationCancelled, check_cancelled
from outfit_studio.utils.logging import log_duration

logger = logging.getLogger(__name__)

StepProgressCallback = Callable[[int, int], None]
PreloadState = Literal["idle", "running", "ready", "failed"]


class InpaintEngine:
    """Lazy-loaded SD inpainting with optional ControlNet (original defaults)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._pipe = None
        self._current_model: str | None = None
        self._use_controlnet = False
        self._architecture: str = "sd15"
        self._warmed_up = False
        self._preload_state: PreloadState = "idle"
        self._preload_error: str | None = None
        self._preload_lock = threading.Lock()
        self._preload_thread: threading.Thread | None = None
        self._work_abort = threading.Event()
        self._model_list_fingerprint: tuple[tuple[str, int], ...] | None = None
        self._model_list_cache: list[dict] | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        logger.info("InpaintEngine ready (device=%s, dtype=%s)", self.device, self.dtype)

    def is_preparing(self) -> bool:
        """True while a background load/compile warmup is in progress."""
        with self._preload_lock:
            return self._preload_state == "running"

    def request_abort(self) -> None:
        """Signal load/compile/warmup to stop at the next checkpoint."""
        self._work_abort.set()

    def clear_work_abort(self) -> None:
        self._work_abort.clear()

    def checkpoint(self) -> None:
        """Raise OperationCancelled when Stop was requested."""
        if self._work_abort.is_set():
            raise OperationCancelled
        check_cancelled()

    def start_background_preload(
        self,
        model_id: str | None = None,
        use_controlnet: bool | None = None,
    ) -> None:
        """Load and warm up the inpaint pipeline on a background thread."""
        with self._preload_lock:
            if self._preload_state in ("running", "ready"):
                return
            if self.device.type != "cuda":
                self._preload_state = "ready"
                return
            self._preload_state = "running"
            self._preload_error = None

        def worker() -> None:
            self.clear_work_abort()
            try:
                logger.info("Background inpaint preload started")
                self.load(model_id, use_controlnet)
                self.warmup()
                with self._preload_lock:
                    self._preload_state = "ready"
                logger.info("Background inpaint preload finished")
            except OperationCancelled:
                logger.info("Background inpaint preload cancelled")
                self.unload()
                with self._preload_lock:
                    self._preload_state = "idle"
            except Exception as exc:
                logger.exception("Background inpaint preload failed")
                with self._preload_lock:
                    self._preload_state = "failed"
                    self._preload_error = str(exc)
            finally:
                self.clear_work_abort()

        thread = threading.Thread(
            target=worker,
            name="inpaint-preload",
            daemon=True,
        )
        with self._preload_lock:
            self._preload_thread = thread
        thread.start()

    def invalidate_model_list_cache(self) -> None:
        self._model_list_fingerprint = None
        self._model_list_cache = None

    def _models_dir_fingerprint(self) -> tuple[tuple[str, int], ...]:
        models_dir = self.settings.resolved_models_dir
        if not models_dir.is_dir():
            return ()
        entries: list[tuple[str, int]] = []
        for pattern in ("*.safetensors", "*.ckpt"):
            for path in models_dir.glob(pattern):
                try:
                    entries.append((path.name, path.stat().st_mtime_ns))
                except OSError:
                    continue
        return tuple(sorted(entries))

    def _discover_local_models(self) -> list[str]:
        models_dir = self.settings.resolved_models_dir
        if not models_dir.is_dir():
            logger.debug("No models directory at %s", models_dir)
            return []
        found: list[str] = []
        for pattern in ("*.safetensors", "*.ckpt"):
            for path in sorted(models_dir.glob(pattern)):
                if inpaint_checkpoint_listable(path):
                    found.append(path.name)
        logger.debug("Discovered %d local checkpoint(s)", len(found))
        return found

    def list_models(self) -> list[dict]:
        fingerprint = self._models_dir_fingerprint()
        if self._model_list_cache is not None and self._model_list_fingerprint == fingerprint:
            return self._model_list_cache

        models: list[dict] = []
        local = self._discover_local_models()
        local_set = set(local)
        default_id = get_default_inpaint_model()
        all_names = list(local)
        if default_id not in all_names:
            all_names.insert(0, default_id)
        for name in get_checkpoint_urls():
            if name not in all_names:
                all_names.append(name)

        for name in all_names:
            if is_hub_model_id(name):
                models.append(
                    {
                        "id": name,
                        "name": name.split("/")[-1],
                        "source": "hub",
                        "arch": "sd15",
                    }
                )
                continue
            path = self._resolve_local_model(name)
            valid = True if name in local_set else path.is_file() and inpaint_checkpoint_valid(path)
            arch = checkpoint_architecture(name, path) if valid else "sd15"
            source = "local" if valid else "download"
            models.append(
                {
                    "id": name,
                    "name": Path(name).stem,
                    "source": source,
                    "arch": arch,
                }
            )

        if not models:
            preferred = self.settings.content.default_inpaint
            models.append(
                {
                    "id": preferred,
                    "name": preferred.split("/")[-1]
                    if is_hub_model_id(preferred)
                    else Path(preferred).stem,
                    "source": "hub" if is_hub_model_id(preferred) else "download",
                    "arch": "sd15",
                }
            )
        self._model_list_fingerprint = fingerprint
        self._model_list_cache = models
        return models

    def default_model_id(self) -> str:
        models = self.list_models()
        preferred = self.settings.content.default_inpaint
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
        if is_hub_model_id(model_id):
            return "sd15"
        path = self._resolve_local_model(model_id)
        if path.is_file() and inpaint_checkpoint_valid(path):
            return checkpoint_architecture(model_id, path)
        return "sd15"

    def download_model(self, model_path: Path) -> Path:
        if model_path.is_file():
            if inpaint_checkpoint_valid(model_path):
                return model_path
            logger.warning("Removing corrupt checkpoint %s", model_path.name)
            model_path.unlink()

        url = get_checkpoint_urls().get(model_path.name)
        if not url:
            msg = f"Model {model_path.name} not found locally and has no download URL"
            raise FileNotFoundError(msg)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s from %s", model_path.name, url)
        response = requests.get(
            url,
            stream=True,
            timeout=HTTP_DOWNLOAD_TIMEOUT_S,
            headers={"User-Agent": HTTP_USER_AGENT},
        )
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            model_path.open("wb") as f,
            tqdm(
                desc=model_path.name,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=BYTES_PER_KIB,
            ) as bar,
        ):
            for chunk in response.iter_content(HTTP_DOWNLOAD_CHUNK_BYTES):
                size = f.write(chunk)
                bar.update(size)

        actual = model_path.stat().st_size
        if total and actual < total * DOWNLOAD_SIZE_TOLERANCE:
            model_path.unlink()
            msg = (
                f"Download of {model_path.name} incomplete "
                f"({actual / BYTES_PER_MIB:.1f} MB of {total / BYTES_PER_MIB:.1f} MB)"
            )
            raise RuntimeError(msg)
        if not inpaint_checkpoint_valid(model_path):
            model_path.unlink()
            msg = f"Downloaded {model_path.name} is not a valid checkpoint"
            raise RuntimeError(msg)
        clear_checkpoint_cache()
        self.invalidate_model_list_cache()
        return model_path

    def _resolve_local_model(self, model_id: str) -> Path:
        primary = self.settings.resolved_models_dir / model_id
        if primary.is_file() and inpaint_checkpoint_valid(primary):
            return primary
        for alias in get_model_aliases().get(model_id, []):
            alias_path = self.settings.resolved_models_dir / alias
            if alias_path.is_file() and inpaint_checkpoint_valid(alias_path):
                return alias_path
        return primary

    def _resolve_model_path(self, model_id: str) -> str:
        if is_hub_model_id(model_id):
            return model_id
        local = self._resolve_local_model(model_id)
        if local.is_file() and inpaint_checkpoint_valid(local):
            return str(local)
        self.download_model(local)
        return str(local)

    def unload(self) -> None:
        if self._pipe is not None:
            logger.info("Unloading inpaint pipeline (model=%s)", self._current_model)
            del self._pipe
            self._pipe = None
            self._warmed_up = False
            free_cuda_cache()

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self, model_id: str | None = None, use_controlnet: bool | None = None) -> None:
        model_id = model_id or self.default_model_id()
        arch = self.model_architecture(model_id)
        use_controlnet = (
            use_controlnet if use_controlnet is not None else self.settings.content.use_controlnet
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

        self.checkpoint()
        self.unload()
        model_path = self._resolve_model_path(model_id)
        logger.info(
            "Loading inpaint model: %s arch=%s controlnet=%s",
            model_path,
            arch,
            use_controlnet,
        )

        with model_load_lock():
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
                            self.settings.content.controlnet,
                            torch_dtype=self.dtype,
                        )
                        if is_hub_model_id(model_path):
                            self._pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                                model_path,
                                controlnet=controlnet,
                                torch_dtype=self.dtype,
                                safety_checker=None,
                            )
                        else:
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

                self.checkpoint()
                assert self._pipe is not None
                if hasattr(self._pipe, "safety_checker"):
                    self._pipe.safety_checker = None
                if hasattr(self._pipe, "set_progress_bar_config"):
                    self._pipe.set_progress_bar_config(disable=True)
                # Deterministic DPM++ is faster per step than SDE and works well with fewer steps.
                self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    self._pipe.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="dpmsolver++",
                )
                self._pipe = self._pipe.to(self.device)
                self._enable_fast_attention(self._pipe)
                if (
                    self.device.type == "cuda"
                    and self.settings.torch_compile
                    and self.settings.torch_compile_cache
                ):
                    load_artifacts(
                        self.settings.resolved_torch_compile_cache_dir,
                        model_id,
                        arch,
                        use_controlnet,
                    )
                self._optimize_for_inference(self._pipe)

        self._current_model = model_id
        self._use_controlnet = use_controlnet
        self._architecture = arch
        logger.info("Inpaint pipeline loaded on %s", self.device)

    def _truncate_prompts(self, prompt: str, negative_prompt: str) -> tuple[str, str]:
        """Keep prompts within CLIP's 77-token limit."""
        if self._pipe is None or not hasattr(self._pipe, "tokenizer"):
            return prompt, negative_prompt
        tokenizer = self._pipe.tokenizer
        max_len = getattr(tokenizer, "model_max_length", CLIP_MAX_TOKENS)

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

    def _optimize_for_inference(self, pipe) -> None:
        """Apply layout and compile optimizations that speed up steady-state inference."""
        if self.device.type != "cuda":
            return

        self.checkpoint()
        for name, module in (
            ("unet", getattr(pipe, "unet", None)),
            ("controlnet", getattr(pipe, "controlnet", None)),
            ("vae", getattr(pipe, "vae", None)),
        ):
            if module is None:
                continue
            try:
                module.to(memory_format=torch.channels_last)
                logger.debug("%s: channels_last layout enabled", name)
            except Exception as exc:
                logger.debug("%s: channels_last skipped (%s)", name, exc)

        if not self.settings.torch_compile:
            logger.info("torch.compile disabled (OUTFIT_STUDIO_TORCH_COMPILE=false)")
            return

        self.checkpoint()
        # ControlNet + reduce-overhead/cudagraphs triggers inductor assertion failures
        # in diffusers pipelines; compile the UNet only with the safer default mode.
        unet = getattr(pipe, "unet", None)
        if unet is None:
            return
        try:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraph_trees = False
        except Exception:
            pass
        try:
            pipe.unet = torch.compile(unet, mode="default", dynamic=False)
            logger.info("torch.compile enabled for unet (mode=default, dynamic=False)")
        except Exception as exc:
            logger.warning("torch.compile failed for unet: %s", exc)
        self.checkpoint()

    @staticmethod
    def _decompile_pipe(pipe) -> bool:
        """Restore eager modules if torch.compile wrappers are present."""
        changed = False
        for name in ("unet", "controlnet"):
            module = getattr(pipe, name, None)
            if module is not None and hasattr(module, "_orig_mod"):
                setattr(pipe, name, module._orig_mod)
                changed = True
        return changed

    @staticmethod
    def _is_compile_runtime_error(exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        markers = ("cudagraph", "assertionerror", "inductor", "dynamo")
        return any(marker in text for marker in markers)

    def warmup(self) -> None:
        """Run a tiny inpaint so CUDA kernels and compile graphs are ready."""
        if self._pipe is None or self.device.type != "cuda" or self._warmed_up:
            return

        self.checkpoint()
        size = self.settings.compile_inpaint_size
        dummy = Image.new("RGB", (size, size), color=(128, 128, 128))
        mask = Image.new("L", (size, size), color=0)
        # Small central mask — enough to exercise the inpaint path without meaningful compute.
        mask.paste(MASK_ON, (size // 4, size // 4, 3 * size // 4, 3 * size // 4))

        logger.info("Warming up inpaint pipeline (%dx%d, 1 step) …", size, size)
        control_image = None
        if self._use_controlnet:
            control_image = Image.new("RGB", (size, size), color=(0, 0, 0))
        try:
            with log_duration(logger, "inpaint warmup"):
                self.inpaint(
                    dummy,
                    mask,
                    prompt="photo",
                    negative_prompt="blur",
                    steps=1,
                    control_image=control_image,
                )
        except OperationCancelled:
            raise
        except Exception as exc:
            logger.warning("Inpaint warmup skipped (%s)", exc)
            return
        self.checkpoint()
        self._warmed_up = True
        if (
            self.settings.torch_compile
            and self.settings.torch_compile_cache
            and self._current_model is not None
        ):
            save_artifacts(
                self.settings.resolved_torch_compile_cache_dir,
                self._current_model,
                self._architecture,
                self._use_controlnet,
            )
        free_cuda_cache()
        logger.info("Inpaint pipeline warm")

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
        strength: float = 1.0,
        on_step: StepProgressCallback | None = None,
    ) -> Image.Image:
        if self._pipe is None:
            self.load()
        assert self._pipe is not None

        steps = steps or self.settings.content.steps
        guidance_scale = guidance_scale or self.settings.content.guidance_scale

        prompt, negative_prompt = self._truncate_prompts(prompt, negative_prompt)

        orig_w, orig_h = image.size
        infer_size = self.settings.compile_inpaint_size
        logger.info(
            "Inpaint %dx%d → %dx%d | steps=%d cfg=%.1f controlnet=%s",
            orig_w,
            orig_h,
            infer_size,
            infer_size,
            steps,
            guidance_scale,
            self._use_controlnet,
        )

        kwargs: dict = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "mask_image": mask,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "width": infer_size,
            "height": infer_size,
            "strength": strength,
        }

        if self._use_controlnet:
            if control_image is None:
                control_image = Image.new("RGB", image.size, color=(0, 0, 0))
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
            try:
                result = self._pipe(**kwargs).images[0]
            except (AssertionError, RuntimeError) as exc:
                if not self._is_compile_runtime_error(exc) or not self._decompile_pipe(self._pipe):
                    raise
                logger.warning(
                    "torch.compile inference failed (%s) — retrying with eager UNet",
                    exc,
                )
                result = self._pipe(**kwargs).images[0]

        if result.size != (orig_w, orig_h):
            result = result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        return result


@lru_cache
def get_inpaint_engine() -> InpaintEngine:
    return InpaintEngine()
