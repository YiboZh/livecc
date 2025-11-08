from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Mapping

import torch
from transformers import logging

logger = logging.get_logger(__name__)

DEFAULT_ALLOCATOR_CONFIG = "expandable_segments:True,max_split_size_mb:512"
DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "auto": None,
    None: None,
}


def apply_runtime_env_defaults(*, fail_fast_ddp: bool = True) -> None:
    """Set reasonable defaults for CUDA allocator and NCCL fail-fast handling."""

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_ALLOCATOR_CONFIG)
    if fail_fast_ddp:
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        os.environ.setdefault("NCCL_DEBUG", os.environ.get("NCCL_DEBUG", "WARN"))


def run_memory_preflight_check(
    model,
    sample_batch: Mapping[str, Any],
    device: torch.device,
    *,
    dtype: str | None = "bfloat16",
    safety_margin_gib: float = 6.0,
    return_to_cpu: bool = True,
) -> tuple[bool, int, int]:
    """Run a forward/backward pass to estimate peak memory usage.

    Returns:
        fits (bool): Whether peak + safety margin fits on device.
        peak_bytes (int): Reported peak allocated CUDA memory.
        total_bytes (int): Total memory available on the device.
    """

    if device.type != "cuda":
        raise ValueError("Memory preflight requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for memory preflight.")
    if sample_batch is None:
        raise ValueError("Sample batch is required for memory preflight.")

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    dtype_obj = _resolve_autocast_dtype(dtype)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=dtype_obj) if dtype_obj else nullcontext()
    )

    original_device = _infer_model_device(model)
    model.train()
    model.to(device)

    batch_on_device = _move_batch_to_device(sample_batch, device)
    with autocast_context:
        outputs = model(**batch_on_device)
        loss = _extract_loss(outputs)
    if loss is None:
        raise RuntimeError("Unable to infer loss from model outputs during preflight.")
    if loss.dim() != 0:
        loss = loss.mean()
    loss.backward()
    torch.cuda.synchronize(device)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    fits = peak_bytes + int(safety_margin_gib * (1024**3)) < total_bytes

    model.zero_grad(set_to_none=True)
    if return_to_cpu:
        target = original_device if original_device is not None else torch.device("cpu")
        model.to(target)
        torch.cuda.empty_cache()
    return fits, peak_bytes, total_bytes


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, Mapping):
            return {k: move(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [move(v) for v in value]
        return value

    return {k: move(v) for k, v in batch.items()}


def _resolve_autocast_dtype(name: str | None):
    if isinstance(name, torch.dtype) or name is None:
        return name
    name_lower = name.lower()
    if name_lower not in DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype for memory check: {name}")
    return DTYPE_ALIASES[name_lower]


def _extract_loss(outputs):
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    return outputs if torch.is_tensor(outputs) else None


def _infer_model_device(model) -> torch.device | None:
    try:
        first_param = next(model.parameters())
        return first_param.device
    except StopIteration:
        return None
