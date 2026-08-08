"""TensorTorrent compile/load helpers for oversized PyTorch modules.

Outfit Studio keeps Diffusers/Transformers orchestration in Python and hands the
heavy ``nn.Module`` bodies (UNet, human parser) to TensorTorrent so
parameters can stream and activations can spill when they exceed VRAM/RAM.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_TT_CACHE_VERSION = "v6"


def tensor_torrent_available() -> bool:
    try:
        import tensortorrent  # noqa: F401

        return True
    except Exception as exc:  # pragma: no cover - import environment
        logger.debug("TensorTorrent unavailable: %s", exc)
        return False


def artifact_ready(artifact_dir: Path) -> bool:
    """True only when load path can open ``exported.pt2`` (not portable-only dirs)."""
    artifact_dir = Path(artifact_dir)
    return (artifact_dir / "compile_config.json").is_file() and (
        artifact_dir / "exported.pt2"
    ).is_file()


def artifact_dir_for(
    cache_root: Path,
    *,
    component: str,
    model_id: str,
    shape_key: str,
) -> Path:
    """Stable per-component artifact path (model + shapes + torch + TT version)."""
    import tensortorrent as tt

    tt_ver = getattr(tt, "__version__", "unknown")
    raw = (
        f"{_TT_CACHE_VERSION}|tt={tt_ver}|torch={torch.__version__}"
        f"|component={component}|model={model_id}|shapes={shape_key}"
    )
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(model_id).name)[:48] or "model"
    return Path(cache_root) / component / f"{slug}_{digest}"


def latency_compile_config(*, cache_dir: Path | None = None) -> Any:
    """Fast path: resident GPU plan, direct path, region torch.compile.

    Measured (SD-sized UNet, RTX 3070 Laptop): MEMORY+stream ~1.85× eager;
    LATENCY no-stream ~1.03×; LATENCY+torch.compile ~0.83×. Diffusion calls UNet
    every step — schedule/transfer tax from MEMORY streaming dominates wall time.
    """
    from tensortorrent.config import CompileConfig, Objective

    has_cuda = torch.cuda.is_available()
    kwargs: dict[str, Any] = {
        "objective": Objective.LATENCY,
        "allow_cpu": not has_cuda,
        "allow_gpu": has_cuda,
        "allow_nvme_streaming": False,
        "allow_host_staged_transfers": True,
        "use_torch_compile": True,
        "prefer_direct_path": True,
        "measure_regions": False,
        "activation_overflow_policy": "spill",
        "prefetch_distance": 1,
        "adaptive_prefetch": True,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = Path(cache_dir)
    return CompileConfig(**kwargs)


def streaming_compile_config(*, cache_dir: Path | None = None) -> Any:
    """Capacity path: MEMORY + NVMe streaming when the model cannot stay resident."""
    from tensortorrent.config import CompileConfig, Objective

    has_cuda = torch.cuda.is_available()
    kwargs: dict[str, Any] = {
        "objective": Objective.MEMORY,
        "allow_cpu": not has_cuda,
        "allow_gpu": has_cuda,
        "allow_nvme_streaming": True,
        "allow_host_staged_transfers": True,
        # Multi-region stream + Dynamo nested compile is fragile; keep FX eager.
        "use_torch_compile": False,
        "prefer_direct_path": True,
        "measure_regions": False,
        "activation_overflow_policy": "spill",
        "prefetch_distance": 2,
        "adaptive_prefetch": True,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = Path(cache_dir)
    return CompileConfig(**kwargs)


def module_param_bytes(module: nn.Module) -> int:
    """Total parameter storage bytes (ignores buffers)."""
    total = 0
    for param in module.parameters():
        total += param.numel() * param.element_size()
    return int(total)


def should_use_tensor_torrent(
    module: nn.Module,
    *,
    min_params_gb: float,
    component: str = "",
) -> bool:
    """Return True only when ``module`` is large enough that TT capacity helps.

    SD1.5 UNet (~1.7 GiB bf16) and the human parser are *small*: TT compile +
    schedule/transfer tax makes them slower than eager/torch.compile, and a
    resident LATENCY plan OOMs next to ControlNet on 8 GB cards. Gate TT to
    models whose params alone exceed ``min_params_gb``.
    """
    params_gb = module_param_bytes(module) / float(1024**3)
    if params_gb < float(min_params_gb):
        logger.info(
            "TensorTorrent skip %s — %.2f GiB params < %.2f GiB threshold (eager / torch.compile)",
            component or type(module).__name__,
            params_gb,
            min_params_gb,
        )
        return False
    return True


def close_compiled(module: Any) -> None:
    closer = getattr(module, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("TensorTorrent close failed: %s", exc)


class CompiledModuleAdapter(nn.Module):
    """Drop-in wrapper: TT owns placement; Diffusers/HF keep calling ``forward``."""

    def __init__(
        self,
        compiled: nn.Module,
        *,
        name: str,
        config: Any | None = None,
        call: Callable[..., Any] | None = None,
        passthrough_attrs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.compiled = compiled
        self.component_name = name
        self.config = config
        self._call = call
        if passthrough_attrs:
            for key, value in passthrough_attrs.items():
                setattr(self, key, value)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._call is not None:
            return self._call(self.compiled, *args, **kwargs)
        return self.compiled(*args, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> CompiledModuleAdapter:  # noqa: ANN401
        return self

    def cpu(self) -> CompiledModuleAdapter:
        return self

    def cuda(self, *args: Any, **kwargs: Any) -> CompiledModuleAdapter:  # noqa: ANN401
        return self

    def close(self) -> None:
        close_compiled(self.compiled)


def strip_export_device_asserts(exported: Any) -> int:
    """Remove ``aten._assert_tensor_metadata`` nodes from an ExportedProgram.

    ``torch.export`` bakes the *capture* device into these asserts (Diffusers
    ``get_time_embed`` / reshape paths). TensorTorrent then schedules Compute on
    CUDA while the assert still expects ``cpu`` (or the reverse) → hard fail.
    They are side-effect-only checks; TT already validates input shapes/dtypes.
    """
    gm = getattr(exported, "graph_module", None)
    if gm is None:
        return 0
    removed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if "assert_tensor_metadata" not in str(node.target):
            continue
        gm.graph.erase_node(node)
        removed += 1
    if removed:
        gm.graph.lint()
        gm.recompile()
        logger.info("Stripped %d _assert_tensor_metadata node(s) from export", removed)
    return removed


@contextmanager
def export_safe_unet_time_embed(module: nn.Module) -> Iterator[None]:
    """Patch Diffusers ``get_time_embed`` so export never traces ``.to(device)``.

    Caller must pass timestep already shaped ``(B,)`` on ``sample.device``
    (see :func:`prepare_unet_timestep` / :func:`unet_example_inputs`).
    """
    target = module
    inner = getattr(module, "inner", None)
    if inner is not None and hasattr(inner, "get_time_embed"):
        target = inner
    if not hasattr(target, "get_time_embed") or not hasattr(target, "time_proj"):
        yield
        return

    original = target.get_time_embed

    def _safe(_self: Any, sample: torch.Tensor, timestep: Any) -> torch.Tensor:
        t = timestep
        if not torch.is_tensor(t):
            t = torch.tensor([int(t)], dtype=torch.int64, device=sample.device)
        elif t.ndim == 0:
            t = t.reshape(1)
        elif t.ndim > 1:
            t = t.reshape(-1)
        if t.shape[0] != sample.shape[0]:
            t = t.expand(sample.shape[0])
        # No ``.to(sample.device)`` — that bakes device asserts into the export.
        t_emb = target.time_proj(t)
        return t_emb.to(dtype=sample.dtype)

    target.get_time_embed = _safe.__get__(target, type(target))
    try:
        yield
    finally:
        target.get_time_embed = original


def _specialize_exported(
    exported: Any,
    *,
    name: str,
    artifact_dir: Path,
    devices: str,
    persist: bool,
    pack_lookup_dirs: tuple[Path, ...] | None = None,
) -> Any:
    """Specialize with LATENCY first; fall back to MEMORY streaming on capacity errors."""
    import tensortorrent as tt
    from tensortorrent.compile.pipeline import compile_exported_program
    from tensortorrent.errors import MemoryCapacityError
    from tensortorrent.frontend.export import _apply_device_selection

    cache_parent = artifact_dir.parent
    configs = (
        ("latency", latency_compile_config(cache_dir=cache_parent)),
        ("streaming", streaming_compile_config(cache_dir=cache_parent)),
    )
    last_exc: BaseException | None = None
    for label, raw_cfg in configs:
        cfg = _apply_device_selection(raw_cfg, devices)
        try:
            logger.info("TensorTorrent specialize %s (%s)", name, label)
            if persist and pack_lookup_dirs is None:
                # Fresh compile: public API persists packs + export.
                return tt.compile_exported(
                    exported,
                    config=raw_cfg,
                    artifact_dir=artifact_dir,
                    devices=devices,
                    name=name,
                )
            compiled = compile_exported_program(
                exported,
                config=cfg,
                name=name,
                artifact_dir=artifact_dir if persist else None,
                pack_lookup_dirs=pack_lookup_dirs,
            )
            if persist and pack_lookup_dirs is not None:
                # Reload path with stripped export — rewrite specialized plan only.
                compiled.save(artifact_dir)
            return compiled
        except MemoryCapacityError as exc:
            last_exc = exc
            logger.warning(
                "TensorTorrent %s %s plan out of capacity (%s)",
                name,
                label,
                exc,
            )
            continue
    assert last_exc is not None
    raise last_exc


def compile_or_load_module(
    module: nn.Module,
    *,
    example_inputs: Any,
    artifact_dir: Path,
    name: str,
    config: Any | None = None,
    call: Callable[..., Any] | None = None,
    passthrough_attrs: dict[str, Any] | None = None,
    strict: bool = False,
) -> CompiledModuleAdapter:
    """Load a cached TT artifact or capture+compile ``module`` into one.

    ``strict=False`` (default) uses ``torch.export`` non-strict mode via
    ``tt.capture_module(..., strict=False)`` so Diffusers-style graphs that need
    graph breaks can still export. Specialize prefers LATENCY (fast); falls back
    to MEMORY streaming only on ``MemoryCapacityError``.
    """
    import tensortorrent as tt

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    devices = "gpu" if torch.cuda.is_available() else "cpu"

    if artifact_ready(artifact_dir):
        logger.info("TensorTorrent load %s ← %s (devices=%s)", name, artifact_dir, devices)
        from tensortorrent.artifact_io import verify_integrity_manifest

        verify_integrity_manifest(artifact_dir, required=True)
        exported = torch.export.load(artifact_dir / "exported.pt2")
        stripped = strip_export_device_asserts(exported)
        if stripped == 0:
            try:
                compiled = tt.load_compiled(artifact_dir)
            except Exception as exc:
                logger.info(
                    "TensorTorrent load_compiled miss for %s (%s) — re-specialize",
                    name,
                    exc,
                )
                compiled = _specialize_exported(
                    exported,
                    name=name,
                    artifact_dir=artifact_dir,
                    devices=devices,
                    persist=False,
                    pack_lookup_dirs=(artifact_dir,),
                )
        else:
            # Persist stripped export + plan so the next boot can load_compiled.
            compiled = _specialize_exported(
                exported,
                name=name,
                artifact_dir=artifact_dir,
                devices=devices,
                persist=True,
                pack_lookup_dirs=(artifact_dir,),
            )
            with suppress(Exception):
                torch.export.save(exported, artifact_dir / "exported.pt2")
    else:
        logger.info(
            "TensorTorrent compile %s → %s (capture strict=%s, devices=%s)",
            name,
            artifact_dir,
            strict,
            devices,
        )
        exported = tt.capture_module(module.eval(), example_inputs, strict=strict)
        strip_export_device_asserts(exported)
        compiled = _specialize_exported(
            exported,
            name=name,
            artifact_dir=artifact_dir,
            devices=devices,
            persist=True,
        )

    return CompiledModuleAdapter(
        compiled,
        name=name,
        config=config,
        call=call,
        passthrough_attrs=passthrough_attrs,
    )


def try_compile_or_load(
    module: nn.Module,
    *,
    example_inputs: Any,
    artifact_dir: Path,
    name: str,
    config: Any | None = None,
    call: Callable[..., Any] | None = None,
    passthrough_attrs: dict[str, Any] | None = None,
    strict: bool = False,
) -> CompiledModuleAdapter | None:
    """Best-effort TT compile/load; returns ``None`` and logs on failure."""
    if not tensor_torrent_available():
        return None
    try:
        return compile_or_load_module(
            module,
            example_inputs=example_inputs,
            artifact_dir=artifact_dir,
            name=name,
            config=config,
            call=call,
            passthrough_attrs=passthrough_attrs,
            strict=strict,
        )
    except Exception as exc:
        logger.warning("TensorTorrent %s failed (%s) — keeping eager module", name, exc)
        return None


def _module_param_device(module: nn.Module) -> torch.device | None:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


# --- Capture wrappers (export-friendly single tensor outs) --------------------


class _TensorOutModule(nn.Module):
    """Wrap a module so ``torch.export`` sees a pure tensor forward."""

    def __init__(self, inner: nn.Module, runner: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.inner = inner
        self._runner = runner

    def forward(self, *args: Any) -> torch.Tensor:
        return self._runner(self.inner, *args)


def unet_is_sdxl(unet: nn.Module) -> bool:
    cfg = getattr(unet, "config", None)
    if cfg is None:
        return False
    if getattr(cfg, "addition_embed_type", None):
        return True
    cross = int(getattr(cfg, "cross_attention_dim", 0) or 0)
    return cross >= 2048


def sdxl_added_cond_dims(unet: nn.Module) -> tuple[int, int]:
    """Return ``(text_embeds_dim, time_ids_len)`` for SDXL ``text_time`` embeds.

    Defaults match Diffusers SDXL: pooled text = 1280, micro-conditioning = 6
    (original size + crop + target size). Aesthetics-score variants use length 5.
    """
    cfg = getattr(unet, "config", None)
    time_ids_len = 5 if bool(getattr(cfg, "requires_aesthetics_score", False)) else 6
    time_embed_dim = int(getattr(cfg, "addition_time_embed_dim", 256) or 256)
    proj_in = int(getattr(cfg, "projection_class_embeddings_input_dim", 2816) or 2816)
    text_dim = max(proj_in - time_ids_len * time_embed_dim, 1280)
    return text_dim, time_ids_len


def unet_example_inputs(
    unet: nn.Module,
    *,
    latent_side: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Build CFG-style (batch=2) example tensors matching an inpaint UNet.

    Timestep is **pre-expanded to ``(B,)`` on ``device``. Diffusers passes a
    scalar ``t``; exporting the UNet's ``t[None].to(sample.device)`` path bakes
    fragile CUDA device asserts that break under TensorTorrent/fake-tensor.

    SDXL returns five tensors: sample, timestep, encoder_hidden_states,
    text_embeds, time_ids. SD1.5 returns the first three only.
    """
    cfg = getattr(unet, "config", None)
    in_channels = int(getattr(cfg, "in_channels", 9) or 9)
    cross_dim = int(getattr(cfg, "cross_attention_dim", 768) or 768)
    sample = torch.randn(2, in_channels, latent_side, latent_side, device=device, dtype=dtype)
    timestep = torch.tensor([999, 999], device=device, dtype=torch.long)  # (B,)
    encoder_hidden_states = torch.randn(2, 77, cross_dim, device=device, dtype=dtype)
    if unet_is_sdxl(unet):
        text_dim, time_ids_len = sdxl_added_cond_dims(unet)
        text_embeds = torch.randn(2, text_dim, device=device, dtype=dtype)
        time_ids = torch.zeros(2, time_ids_len, device=device, dtype=dtype)
        return sample, timestep, encoder_hidden_states, text_embeds, time_ids
    return sample, timestep, encoder_hidden_states


def normalize_unet_timestep(timestep: Any, *, device: torch.device | None = None) -> torch.Tensor:
    """Coerce Diffusers timestep forms to a long tensor (0-dim or 1-d)."""
    if isinstance(timestep, torch.Tensor):
        t = timestep.detach()
        return (
            t.to(device=device, dtype=torch.long) if device is not None else t.to(dtype=torch.long)
        )
    return torch.tensor(int(timestep), dtype=torch.long, device=device)


def prepare_unet_timestep(sample: torch.Tensor, timestep: Any) -> torch.Tensor:
    """Match Diffusers ``get_time_embed`` batching without tracing ``.to(sample.device)``.

    Returns shape ``(N,)`` on ``sample.device`` (N = sample batch).
    """
    t = normalize_unet_timestep(timestep, device=sample.device)
    if t.ndim == 0:
        t = t[None]
    elif t.ndim > 1:
        t = t.reshape(-1)
    if t.shape[0] == 1 and sample.shape[0] > 1:
        t = t.expand(sample.shape[0])
    elif t.shape[0] != sample.shape[0]:
        t = t.reshape(-1)[0].expand(sample.shape[0])
    return t.contiguous()


def _run_unet(
    inner: nn.Module,
    sample,
    timestep,
    encoder_hidden_states,
    text_embeds: torch.Tensor | None = None,
    time_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "encoder_hidden_states": encoder_hidden_states,
        "return_dict": False,
    }
    if text_embeds is not None and time_ids is not None:
        kwargs["added_cond_kwargs"] = {
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
    return inner(sample, timestep, **kwargs)[0]


def _unet_diffusers_call(
    compiled: nn.Module,
    *args: Any,
    expects_sdxl: bool = False,
    **kwargs: Any,
) -> Any:
    sample = args[0] if args else kwargs["sample"]
    timestep = args[1] if len(args) > 1 else kwargs["timestep"]
    encoder_hidden_states = args[2] if len(args) > 2 else kwargs.get("encoder_hidden_states")
    if encoder_hidden_states is None:
        encoder_hidden_states = kwargs["encoder_hidden_states"]
    if not isinstance(sample, torch.Tensor):
        raise TypeError("UNet sample must be a torch.Tensor")
    timestep = prepare_unet_timestep(sample, timestep)
    if isinstance(encoder_hidden_states, torch.Tensor):
        encoder_hidden_states = encoder_hidden_states.to(
            device=sample.device, dtype=encoder_hidden_states.dtype
        )

    added = kwargs.get("added_cond_kwargs") or {}
    text_embeds = added.get("text_embeds") if isinstance(added, dict) else None
    time_ids = added.get("time_ids") if isinstance(added, dict) else None
    if expects_sdxl or (text_embeds is not None and time_ids is not None):
        if text_embeds is None or time_ids is None:
            raise TypeError(
                "SDXL TensorTorrent UNet requires added_cond_kwargs with text_embeds and time_ids"
            )
        if isinstance(text_embeds, torch.Tensor):
            text_embeds = text_embeds.to(device=sample.device, dtype=text_embeds.dtype)
        if isinstance(time_ids, torch.Tensor):
            time_ids = time_ids.to(device=sample.device, dtype=time_ids.dtype)
        out = compiled(sample, timestep, encoder_hidden_states, text_embeds, time_ids)
    else:
        out = compiled(sample, timestep, encoder_hidden_states)
    if kwargs.get("return_dict", True) is False:
        return (out,)
    return SimpleNamespace(sample=out)


def try_compile_unet(
    unet: nn.Module,
    *,
    model_id: str,
    cache_root: Path,
    latent_side: int,
    device: torch.device,
    dtype: torch.dtype,
    enabled: bool = True,
    min_params_gb: float = 4.0,
) -> CompiledModuleAdapter | None:
    """Compile a Diffusers UNet with TensorTorrent when oversized enough.

    Small UNets (SD1.5) skip TT — see :func:`should_use_tensor_torrent`.
    SDXL exports ``text_embeds`` / ``time_ids`` as plain tensors (dict kwargs
    are not torch.export-friendly).
    """
    if not enabled:
        logger.info("TensorTorrent UNet skipped (%s) — disabled by settings", model_id)
        return None
    if not should_use_tensor_torrent(
        unet, min_params_gb=min_params_gb, component=f"unet:{model_id}"
    ):
        return None

    capture_device = (
        device if device.type == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    )
    examples = unet_example_inputs(
        unet, latent_side=latent_side, device=capture_device, dtype=dtype
    )
    sample, enc = examples[0], examples[2]
    sdxl = len(examples) == 5
    shape_key = f"b2_c{sample.shape[1]}_s{latent_side}_d{enc.shape[-1]}_tB_{capture_device.type}"
    if sdxl:
        shape_key += f"_te{examples[3].shape[-1]}_tid{examples[4].shape[-1]}"
    artifact_dir = artifact_dir_for(
        cache_root, component="unet", model_id=model_id, shape_key=shape_key
    )
    # SD1.5: 3 tensors. SDXL: + text_embeds/time_ids as plain args (export-safe).
    core = _TensorOutModule(unet, _run_unet)
    # Keep module on capture_device (do not force CPU).
    origin = _module_param_device(core)
    if origin is not None and origin != capture_device:
        core.to(capture_device)

    with export_safe_unet_time_embed(core):
        compiled = try_compile_or_load(
            core,
            example_inputs=examples,
            artifact_dir=artifact_dir,
            name="unet",
            config=getattr(unet, "config", None),
            call=lambda c, *a, **k: _unet_diffusers_call(c, *a, expects_sdxl=sdxl, **k),
            passthrough_attrs={
                "dtype": dtype,
                "device": device,
                "tt_expects_sdxl_cond": sdxl,
            },
            strict=False,
        )
    if compiled is None and origin is not None and origin != capture_device:
        try:
            core.to(origin)
        except Exception as exc:  # pragma: no cover
            logger.debug("restore unet device after TT fail: %s", exc)
    return compiled


def try_compile_human_parser(
    model: nn.Module,
    *,
    model_id: str,
    cache_root: Path,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    min_params_gb: float = 4.0,
) -> CompiledModuleAdapter | None:
    if not should_use_tensor_torrent(
        model, min_params_gb=min_params_gb, component=f"human_parser:{model_id}"
    ):
        return None
    pixel_values = torch.randn(1, 3, image_size, image_size, device=device, dtype=dtype)
    shape_key = f"b1_3_{image_size}"
    artifact_dir = artifact_dir_for(
        cache_root, component="human_parser", model_id=model_id, shape_key=shape_key
    )

    def _run_parser(inner: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
        return inner(pixel_values=pixels).logits

    def _parser_call(compiled: nn.Module, *args: Any, **kwargs: Any) -> Any:
        pixels = args[0] if args else kwargs.get("pixel_values")
        if pixels is None:
            raise TypeError("human parser expects pixel_values")
        # Safety net: TT rejects dynamic H/W — resize to the compiled square.
        if pixels.ndim != 4:
            raise TypeError(f"pixel_values must be NCHW, got shape {tuple(pixels.shape)}")
        if pixels.shape[-2:] != (image_size, image_size):
            pixels = torch.nn.functional.interpolate(
                pixels.float(),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=pixels.dtype)
        logits = compiled(pixels)
        return SimpleNamespace(logits=logits)

    core = _TensorOutModule(model, _run_parser)
    return try_compile_or_load(
        core,
        example_inputs=(pixel_values,),
        artifact_dir=artifact_dir,
        name="human_parser",
        config=getattr(model, "config", None),
        call=_parser_call,
        passthrough_attrs={"tt_input_size": image_size},
    )
