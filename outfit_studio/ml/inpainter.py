"""Stable Diffusion ControlNet inpainting for clothing edits."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import suppress
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
        self._cpu_offload = False
        # Offload ladder: None (full GPU) → "model" → "sequential" | "tt_partial".
        self._offload_mode: Literal["model", "sequential", "tt_partial"] | None = None
        self._oom_retried = False
        self._deferred_sdxl_offload = False
        self._warmup_active = False
        self._tt_unet_eager: object | None = None
        self._preload_state: PreloadState = "idle"
        self._preload_lock = threading.Lock()
        self._preload_thread: threading.Thread | None = None
        self._preload_done = threading.Event()
        self._preload_done.set()
        self._work_abort = threading.Event()
        self._model_list_fingerprint: tuple[tuple[str, int], ...] | None = None
        self._model_list_cache: list[dict] | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # float16 (not bf16): SDXL VAE is fragile in bf16; Ampere fp16 Tensor Cores are fine.
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        logger.info("InpaintEngine ready (device=%s, dtype=%s)", self.device, self.dtype)

    def is_preparing(self) -> bool:
        """True while a background load/compile warmup is in progress."""
        with self._preload_lock:
            return self._preload_state == "running"

    def wait_for_preload(
        self,
        *,
        progress: Callable[[float, str], None] | None = None,
    ) -> None:
        """Block until background load/compile/warmup completes."""
        while self.is_preparing():
            self.checkpoint()
            if progress is not None:
                progress(0, desc="Loading and compiling model…")
            self._preload_done.wait(timeout=0.25)

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
            self._preload_done.clear()

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
            except Exception:
                logger.exception("Background inpaint preload failed")
                with self._preload_lock:
                    self._preload_state = "failed"
            finally:
                self.clear_work_abort()
                self._preload_done.set()

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

    def _checkpoint_download_headers(self, url: str) -> dict[str, str]:
        headers = {"User-Agent": HTTP_USER_AGENT}
        token = (self.settings.civitai_api_token or "").strip()
        if token and "civitai." in url.lower():
            headers["Authorization"] = f"Bearer {token}"
        return headers

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
            headers=self._checkpoint_download_headers(url),
        )
        if response.status_code == 401:
            msg = (
                f"Unauthorized downloading {model_path.name} from {url}. "
                "Set OUTFIT_STUDIO_CIVITAI_API_TOKEN in .env "
                "(Civitai API key with download permission)."
            )
            raise requests.HTTPError(msg, response=response)
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
            from outfit_studio.ml.tt_runtime import close_compiled

            for name in ("unet", "controlnet", "vae"):
                module = getattr(self._pipe, name, None)
                if module is not None:
                    close_compiled(module)
            del self._pipe
            self._pipe = None
            self._warmed_up = False
            self._cpu_offload = False
            self._offload_mode = None
            self._oom_retried = False
            self._deferred_sdxl_offload = False
            self._tt_unet_eager = None
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
                        vae = self._load_sdxl_vae()
                        sdxl_kwargs: dict = {"torch_dtype": self.dtype}
                        if vae is not None:
                            sdxl_kwargs["vae"] = vae
                        if model_path.endswith(".safetensors"):
                            self._pipe = StableDiffusionXLInpaintPipeline.from_single_file(
                                model_path,
                                **sdxl_kwargs,
                            )
                        else:
                            self._pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                                model_path,
                                **sdxl_kwargs,
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
                # Arch/CN flags must be set before placement (SDXL VAE + size gates).
                self._architecture = arch
                self._use_controlnet = use_controlnet
                self._current_model = model_id
                if hasattr(self._pipe, "safety_checker"):
                    self._pipe.safety_checker = None
                if hasattr(self._pipe, "set_progress_bar_config"):
                    self._pipe.set_progress_bar_config(disable=True)
                # SD1.5: deterministic DPM++ (fast). SDXL cards (Lustify): DPM++ SDE + Karras.
                if arch == "sdxl":
                    self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                        self._pipe.scheduler.config,
                        use_karras_sigmas=True,
                        algorithm_type="sde-dpmsolver++",
                    )
                else:
                    self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                        self._pipe.scheduler.config,
                        use_karras_sigmas=True,
                        algorithm_type="dpmsolver++",
                    )
                self._move_pipe_to_device()
                self._enable_fast_attention(self._pipe)
                if (
                    self.device.type == "cuda"
                    and not self.settings.tensor_torrent
                    and self.settings.torch_compile
                    and self.settings.torch_compile_cache
                ):
                    load_artifacts(
                        self.settings.resolved_torch_compile_cache_dir,
                        model_id,
                        arch,
                        use_controlnet,
                    )
                self._optimize_for_inference(self._pipe, model_id=model_id)

        logger.info("Inpaint pipeline loaded on %s", self.device)

    def inference_size(self) -> int:
        """Square side for inpaint — SDXL needs ≥768 (native UNet sample_size=128 → 1024)."""
        from outfit_studio.constants import LATENT_ALIGN, MIN_LATENT_SIDE, SDXL_MIN_INFER_SIZE

        base = int(self.settings.content.inference_size)
        if self._architecture == "sdxl" or (
            self._current_model and self.model_architecture(self._current_model) == "sdxl"
        ):
            base = max(base, SDXL_MIN_INFER_SIZE)
        aligned = base // LATENT_ALIGN * LATENT_ALIGN
        return max(aligned, MIN_LATENT_SIDE)

    def _move_pipe_to_device(self) -> None:
        """Prefer full GPU residency; escalate offload only after OOM.

        Measured on RTX 3070 Ti 8 GB (fp16, peers freed):
        - SD1.5 + ControlNet full GPU: ~3.3 GiB peak, ~1.5 s / 4 steps
        - model_cpu_offload: ~2× slower
        - sequential_cpu_offload: ~3× slower (last resort; SDXL UNet alone ~4.8 GiB)
        """
        from outfit_studio.ml.gpu_memory import (
            free_cuda_cache,
            gpu_free_gb,
            prepare_for_inpaint,
        )

        assert self._pipe is not None
        if self.device.type == "cuda":
            prepare_for_inpaint()
            free_cuda_cache()
            # SDXL UNet ≈4.8 GiB — keep it on GPU via model offload (one transfer per
            # stage). Sequential layer-offload wastes VRAM headroom (~3 GiB idle) and
            # is much slower. Fall back to sequential only if model-offload OOMs later.
            from outfit_studio.ml.gpu_memory import vram_is_tight

            if self._architecture == "sdxl" and vram_is_tight():
                # Prefer TensorTorrent for oversized UNets before accelerate offload —
                # offload-first skips TT (optimize bails on `_cpu_offload`).
                if self.settings.tensor_torrent and self.settings.tensor_torrent_unet:
                    logger.info("SDXL tight VRAM — defer accelerate offload until TensorTorrent")
                    self._deferred_sdxl_offload = True
                    try:
                        self._pipe = self._pipe.to(self.device)
                        self._cpu_offload = False
                        self._offload_mode = None
                    except torch.OutOfMemoryError:
                        logger.info(
                            "SDXL .to(cuda) OOM before TT — keep weights on CPU for capture"
                        )
                        with suppress(Exception):
                            self._pipe = self._pipe.to("cpu")
                        self._cpu_offload = False
                        self._offload_mode = None
                    self._enable_low_vram_guards(self._pipe, offload=None)
                    self._prepare_sdxl_vae(self._pipe)
                    return
                free = gpu_free_gb()
                # UNet ≈4.8 GiB; model-offload needs ~6+ GiB free for weights+activations.
                # Below that, sequential is the only path that fits (peak ~1–3 GiB).
                if free >= 6.0:
                    logger.info(
                        "SDXL — model CPU offload (free=%.2f GB; UNet stays on GPU in denoise)",
                        free,
                    )
                    mode: Literal["model", "sequential"] = "model"
                else:
                    logger.info(
                        "SDXL — sequential offload (free=%.2f GB < 6 GB needed for "
                        "resident UNet; peak VRAM stays low by design)",
                        free,
                    )
                    mode = "sequential"
                if not self._enable_low_vram_guards(self._pipe, offload=mode):
                    self._enable_low_vram_guards(self._pipe, offload="sequential")
                return
            logger.info(
                "Moving inpaint pipeline → %s (free=%.2f GB, arch=%s, controlnet=%s)",
                self.device,
                gpu_free_gb(),
                self._architecture,
                self._use_controlnet,
            )
        try:
            self._pipe = self._pipe.to(self.device)
            self._cpu_offload = False
            self._offload_mode = None
        except torch.OutOfMemoryError:
            if self.device.type != "cuda":
                raise
            logger.warning("Inpaint .to(cuda) OOM — trying model CPU offload")
            prepare_for_inpaint()
            free_cuda_cache()
            if not self._enable_low_vram_guards(self._pipe, offload="model"):
                logger.warning("model CPU offload unavailable — sequential offload")
                self._enable_low_vram_guards(self._pipe, offload="sequential")
            return
        if self.device.type == "cuda":
            self._enable_low_vram_guards(self._pipe, offload=None)
            self._prepare_sdxl_vae(self._pipe)

    def _load_sdxl_vae(self):
        """FP16-safe SDXL VAE — stock VAE needs fp32 upcast and OOMs model-offload @1024."""
        from diffusers import AutoencoderKL

        try:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=self.dtype,
            )
            logger.info("SDXL VAE: madebyollin/sdxl-vae-fp16-fix (fp16-safe)")
            return vae
        except Exception as exc:
            logger.warning("sdxl-vae-fp16-fix unavailable (%s) — stock VAE + force_upcast", exc)
            return None

    def _prepare_sdxl_vae(self, pipe) -> None:
        """Stock SDXL VAE needs fp32 encode/decode; fp16-fix does not."""
        if self._architecture != "sdxl":
            return
        vae = getattr(pipe, "vae", None)
        if vae is None:
            return
        name = ""
        with suppress(Exception):
            name = str(getattr(getattr(vae, "config", None), "_name_or_path", "") or "")
        if "fp16-fix" in name.lower():
            with suppress(Exception):
                vae.config.force_upcast = False
            return
        with suppress(Exception):
            if hasattr(vae, "config"):
                vae.config.force_upcast = True
            logger.info("SDXL VAE force_upcast=True (fp32 encode/decode)")

    def _enable_low_vram_guards(
        self,
        pipe,
        *,
        offload: Literal["model", "sequential"] | None = None,
    ) -> bool:
        """Attention/VAE slicing on tight cards; optional accelerate offload.

        Returns True when the requested offload mode was installed.
        """
        from outfit_studio.constants import VRAM_TIGHT_TOTAL_GB
        from outfit_studio.ml.gpu_memory import free_cuda_cache, gpu_total_gb, vram_is_tight

        installed = False
        if offload is not None:
            free_cuda_cache()
            with suppress(Exception):
                pipe.to("cpu")
            free_cuda_cache()
            try:
                if offload == "model" and hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                    self._cpu_offload = True
                    self._offload_mode = "model"
                    installed = True
                    logger.info(
                        "enable_model_cpu_offload active (GPU %.1f GB)",
                        gpu_total_gb(),
                    )
                elif offload == "sequential" and hasattr(pipe, "enable_sequential_cpu_offload"):
                    pipe.enable_sequential_cpu_offload()
                    self._cpu_offload = True
                    self._offload_mode = "sequential"
                    installed = True
                    logger.info(
                        "enable_sequential_cpu_offload active (GPU %.1f GB) — slow path",
                        gpu_total_gb(),
                    )
                elif offload == "model" and hasattr(pipe, "enable_sequential_cpu_offload"):
                    # No model-offload API — fall through to sequential.
                    pipe.enable_sequential_cpu_offload()
                    self._cpu_offload = True
                    self._offload_mode = "sequential"
                    installed = True
                    logger.info("enable_sequential_cpu_offload active (no model-offload API)")
                else:
                    raise ImportError("pipeline has no CPU offload helpers")
            except ImportError as exc:
                logger.warning(
                    "accelerate CPU offload unavailable (%s) — partial GPU residency",
                    exc,
                )
                self._partial_gpu_residency(pipe)
                installed = True
            self._prepare_sdxl_vae(pipe)

        if not vram_is_tight() and not self._cpu_offload:
            return installed

        # VAE tiling mottles SDXL decode; keep slicing only.
        slice_methods: list[tuple[str, dict]] = [
            ("enable_attention_slicing", {"slice_size": "max"}),
            ("enable_vae_slicing", {}),
        ]
        if self._architecture != "sdxl":
            slice_methods.append(("enable_vae_tiling", {}))

        for method, call_kwargs in slice_methods:
            fn = getattr(pipe, method, None)
            if not callable(fn):
                continue
            try:
                if call_kwargs:
                    fn(**call_kwargs)
                else:
                    fn()
                logger.info("%s enabled (GPU < %.0f GB)", method, VRAM_TIGHT_TOTAL_GB)
            except TypeError:
                with suppress(Exception):
                    fn()
                    logger.info("%s enabled (GPU < %.0f GB)", method, VRAM_TIGHT_TOTAL_GB)
            except Exception as exc:
                logger.debug("%s skipped: %s", method, exc)
        return installed

    def _partial_gpu_residency(self, pipe) -> None:
        """Fallback without accelerate: UNet (+ ControlNet) on CUDA, rest on CPU."""
        from outfit_studio.ml.gpu_memory import free_cuda_cache

        free_cuda_cache()
        with suppress(Exception):
            pipe.to("cpu")
        free_cuda_cache()
        for name in ("unet", "controlnet"):
            mod = getattr(pipe, name, None)
            if mod is None:
                continue
            try:
                setattr(pipe, name, mod.to(self.device))
                logger.info("%s → %s (partial GPU residency)", name, self.device)
            except torch.OutOfMemoryError:
                logger.warning("%s stayed on CPU — not enough VRAM even alone", name)
                free_cuda_cache()
        self._cpu_offload = True  # treat as offload-ish: skip compile/layout
        self._offload_mode = self._offload_mode or "model"
        # Diffusers encode_prompt needs text_encoder briefly on CUDA.
        self._install_transient_text_encoder_device(pipe)

    def _install_transient_text_encoder_device(self, pipe) -> None:
        """Move text encoders to CUDA only around encode_prompt when they stay on CPU."""
        encode = getattr(pipe, "encode_prompt", None)
        if not callable(encode) or getattr(pipe, "_os_te_wrap", False):
            return
        engine = self
        te_names = ("text_encoder", "text_encoder_2")

        def _wrapped(*args, **kwargs):  # noqa: ANN002
            moved: list[tuple[str, object]] = []
            for name in te_names:
                te = getattr(pipe, name, None)
                if te is None:
                    continue
                try:
                    setattr(pipe, name, te.to(engine.device))
                    moved.append((name, te))
                except torch.OutOfMemoryError:
                    from outfit_studio.ml.gpu_memory import free_cuda_cache

                    free_cuda_cache()
                    setattr(pipe, name, te.to(engine.device))
                    moved.append((name, te))
            try:
                return encode(*args, **kwargs)
            finally:
                if moved:
                    for name, _orig in moved:
                        current = getattr(pipe, name, None)
                        if current is not None:
                            setattr(pipe, name, current.to("cpu"))
                    from outfit_studio.ml.gpu_memory import free_cuda_cache

                    free_cuda_cache()

        pipe.encode_prompt = _wrapped  # type: ignore[method-assign]
        pipe._os_te_wrap = True  # type: ignore[attr-defined]
        logger.info("text_encoder(+_2) transient CUDA wrap installed (partial residency)")

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

    def _optimize_for_inference(self, pipe, *, model_id: str) -> None:
        """Apply layout + TensorTorrent (preferred) or torch.compile optimizations."""
        if self.device.type != "cuda":
            return
        # Accelerate offload owns placement — skip compile unless we deferred for TT.
        if self._cpu_offload and not self._deferred_sdxl_offload:
            logger.info("Skipping compile/layout — model CPU offload owns placement")
            return

        from outfit_studio.ml.gpu_memory import gpu_total_gb, vram_is_tight

        tight = vram_is_tight()
        self.checkpoint()
        if not tight and not self._cpu_offload:
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

        if self.settings.tensor_torrent:
            self.checkpoint()
            if self._apply_tensor_torrent(pipe, model_id=model_id):
                self._finalize_after_tensor_torrent(pipe)
                return
            logger.info("TensorTorrent unavailable for this pipeline — falling back")

        if self._deferred_sdxl_offload:
            self._install_deferred_sdxl_offload(pipe)
            return

        if tight:
            logger.info(
                "torch.compile skipped on tight VRAM (%.1f GB) — keeps activation headroom",
                gpu_total_gb(),
            )
            return

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

    def _install_deferred_sdxl_offload(self, pipe) -> None:
        """Accelerate offload after TT failed on a tight SDXL card."""
        from outfit_studio.ml.gpu_memory import gpu_free_gb

        self._deferred_sdxl_offload = False
        free = gpu_free_gb()
        mode: Literal["model", "sequential"] = "model" if free >= 6.0 else "sequential"
        logger.info("SDXL deferred offload after TT miss → %s (free=%.2f GB)", mode, free)
        if not self._enable_low_vram_guards(pipe, offload=mode):
            self._enable_low_vram_guards(pipe, offload="sequential")

    def _finalize_after_tensor_torrent(self, pipe) -> None:
        """TT owns UNet placement; park peers on tight VRAM (no accelerate on TT UNet)."""
        from outfit_studio.ml.gpu_memory import free_cuda_cache, vram_is_tight

        self._deferred_sdxl_offload = False
        if not vram_is_tight():
            return
        for name in ("vae", "text_encoder", "text_encoder_2", "controlnet"):
            mod = getattr(pipe, name, None)
            if mod is None:
                continue
            with suppress(Exception):
                setattr(pipe, name, mod.to("cpu"))
        free_cuda_cache()
        self._cpu_offload = True
        self._offload_mode = "tt_partial"
        self._install_transient_text_encoder_device(pipe)
        self._enable_low_vram_guards(pipe, offload=None)
        logger.info("TensorTorrent UNet + peer modules on CPU (tt_partial)")

    def _apply_tensor_torrent(self, pipe, *, model_id: str) -> bool:
        """Compile/load UNet via TensorTorrent when enabled.

        ControlNet stays eager (Diffusers ``isinstance`` checks). Temporarily
        parks non-UNet pipe modules on CPU so CUDA capture has VRAM headroom.
        """
        from outfit_studio.constants import LATENT_ALIGN
        from outfit_studio.ml.gpu_memory import free_cuda_cache, vram_is_tight
        from outfit_studio.ml.tt_runtime import try_compile_unet

        cache_root = self.settings.resolved_tensor_torrent_cache_dir
        latent_side = max(self.inference_size() // LATENT_ALIGN, 8)

        unet = getattr(pipe, "unet", None)
        if unet is None:
            return False

        parked: list[tuple[str, object]] = []
        if self.device.type == "cuda":
            for name in ("vae", "text_encoder", "text_encoder_2", "controlnet"):
                mod = getattr(pipe, name, None)
                if mod is None:
                    continue
                try:
                    parked.append((name, mod))
                    setattr(pipe, name, mod.to("cpu"))
                except Exception as exc:
                    logger.debug("park %s for TT capture skipped: %s", name, exc)
            free_cuda_cache()

        try:
            compiled = try_compile_unet(
                unet,
                model_id=model_id,
                cache_root=cache_root,
                latent_side=latent_side,
                device=self.device,
                dtype=self.dtype,
                enabled=self.settings.tensor_torrent_unet,
                min_params_gb=self.settings.tensor_torrent_min_params_gb,
            )
        finally:
            # Tight VRAM: leave peers on CPU (finalize parks again). Else restore to GPU.
            restore_device = torch.device("cpu") if vram_is_tight() else self.device
            for name, mod in parked:
                try:
                    setattr(pipe, name, mod.to(restore_device))
                except Exception as exc:
                    logger.warning("restore %s after TT capture failed: %s", name, exc)
            if parked:
                free_cuda_cache()

        if compiled is None:
            return False
        self._tt_unet_eager = unet
        pipe.unet = compiled
        logger.info("TensorTorrent active for unet (%s); ControlNet remains eager", model_id)
        return True

    def _unwrap_compiled_modules(self, pipe) -> bool:
        """Restore eager UNet/ControlNet from torch.compile or TensorTorrent adapters."""
        from outfit_studio.ml.tt_runtime import CompiledModuleAdapter, close_compiled

        changed = False
        for name in ("unet", "controlnet"):
            module = getattr(pipe, name, None)
            if module is None:
                continue
            if hasattr(module, "_orig_mod"):
                setattr(pipe, name, module._orig_mod)
                changed = True
                continue
            if isinstance(module, CompiledModuleAdapter) or getattr(module, "component_name", None):
                if name == "unet" and self._tt_unet_eager is not None:
                    close_compiled(module)
                    setattr(pipe, name, self._tt_unet_eager)
                    self._tt_unet_eager = None
                    changed = True
                else:
                    logger.warning("Cannot restore eager %s from TensorTorrent (no backup)", name)
        return changed

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
        # Bare AssertionError (empty message) is Diffusers check_inputs, not inductor.
        if isinstance(exc, AssertionError) and not str(exc):
            return False
        text = f"{type(exc).__name__}: {exc}".lower()
        markers = ("cudagraph", "inductor", "dynamo", "accessing tensor output")
        return any(marker in text for marker in markers)

    def warmup(self) -> None:
        """Run a tiny inpaint so CUDA kernels and compile graphs are ready."""
        if self._pipe is None or self.device.type != "cuda" or self._warmed_up:
            return

        self.checkpoint()
        size = self.inference_size()
        dummy = Image.new("RGB", (size, size), color=(128, 128, 128))
        mask = Image.new("L", (size, size), color=0)
        # Small central mask — enough to exercise the inpaint path without meaningful compute.
        mask.paste(MASK_ON, (size // 4, size // 4, 3 * size // 4, 3 * size // 4))

        logger.info("Warming up inpaint pipeline (%dx%d, 1 step) …", size, size)
        control_image = None
        if self._use_controlnet:
            control_image = Image.new("RGB", (size, size), color=(0, 0, 0))
        self._warmup_active = True
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
        finally:
            self._warmup_active = False
        self.checkpoint()
        self._warmed_up = True
        # Eager UNet backup only needed for mid-run OOM escalate.
        self._tt_unet_eager = None
        if (
            not self.settings.tensor_torrent
            and self.settings.torch_compile
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

        if self._architecture == "sdxl" and not self._warmup_active:
            from outfit_studio.constants import (
                SDXL_GUIDANCE_MAX,
                SDXL_INPAINT_STRENGTH,
                SDXL_MIN_STEPS,
            )

            if guidance_scale > SDXL_GUIDANCE_MAX:
                logger.warning(
                    "SDXL CFG %.1f is too high (model cards recommend ≤%.1f) — clamping "
                    "(CFG 10 causes muddy/blob inpaint)",
                    guidance_scale,
                    SDXL_GUIDANCE_MAX,
                )
                guidance_scale = SDXL_GUIDANCE_MAX
            if steps < SDXL_MIN_STEPS:
                logger.info(
                    "SDXL steps %d → %d (Lustify-class cards want ~30)",
                    steps,
                    SDXL_MIN_STEPS,
                )
                steps = SDXL_MIN_STEPS
            if strength > SDXL_INPAINT_STRENGTH:
                logger.info(
                    "SDXL strength %.2f → %.2f (full denoise + soft mask → incoherent fill)",
                    strength,
                    SDXL_INPAINT_STRENGTH,
                )
                strength = SDXL_INPAINT_STRENGTH

        prompt, negative_prompt = self._truncate_prompts(prompt, negative_prompt)

        orig_w, orig_h = image.size
        infer_size = self.inference_size()
        logger.info(
            "Inpaint %dx%d → %dx%d | steps=%d cfg=%.1f controlnet=%s offload=%s",
            orig_w,
            orig_h,
            infer_size,
            infer_size,
            steps,
            guidance_scale,
            self._use_controlnet,
            self._offload_mode or "full_gpu",
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

        # SDXL VAE force_upcast needs real fp32 ops — outer autocast(fp16) fights that.
        use_autocast = self.device.type == "cuda" and self._architecture != "sdxl"
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=use_autocast,
            ),
        ):
            try:
                result = self._pipe(**kwargs).images[0]
            except torch.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                if self._oom_retried or self._offload_mode == "sequential":
                    raise
                if self._offload_mode == "tt_partial":
                    # TT already owns capacity path; accelerate cannot move the adapter.
                    logger.error("Inpaint OOM with TensorTorrent UNet — no accelerate escalate")
                    raise
                self._oom_retried = True
                next_mode: Literal["model", "sequential"] = (
                    "sequential" if self._offload_mode == "model" else "model"
                )
                logger.warning(
                    "Inpaint forward OOM — escalate offload → %s, retry once",
                    next_mode,
                )
                free_cuda_cache()
                self._unwrap_compiled_modules(self._pipe)
                if hasattr(self._pipe, "maybe_free_model_hooks"):
                    with suppress(Exception):
                        self._pipe.maybe_free_model_hooks()
                self._cpu_offload = False
                self._offload_mode = None
                self._enable_low_vram_guards(self._pipe, offload=next_mode)
                free_cuda_cache()
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
