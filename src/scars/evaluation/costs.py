from __future__ import annotations

from time import perf_counter
import platform
import subprocess

import numpy as np


def _nvidia_snapshot() -> dict[str, object] | None:
    query = "driver_version,power.limit,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate"
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "--id=0"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()[0]
        driver, power_limit, temperature, sm_clock, memory_clock, pstate = [item.strip() for item in output.split(",")]
        return {
            "driver_version": driver,
            "power_limit_w": float(power_limit),
            "temperature_c": float(temperature),
            "sm_clock_mhz": float(sm_clock),
            "memory_clock_mhz": float(memory_clock),
            "pstate": pstate,
        }
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def _timing_stability(samples: list[float]) -> dict[str, object]:
    values = np.asarray(samples, dtype=float)
    midpoint = max(1, len(values) // 2)
    first = float(np.median(values[:midpoint]))
    second = float(np.median(values[midpoint:]))
    ratio = max(first, second) / max(min(first, second), 1.0e-12)
    return {"first_half_median_ms": first, "second_half_median_ms": second, "drift_ratio": ratio, "drift_valid": ratio <= 1.25}


def tensor_bytes(shape: tuple[int, ...], dtype_bytes: int = 4) -> int:
    total = dtype_bytes
    for dimension in shape:
        total *= dimension
    return total


def model_parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def estimate_model_macs(model, example) -> int:
    """Count Conv/Linear/attention multiply-accumulates for one example.

    Elementwise activations, normalization, pooling, softmax and indexing are
    deliberately excluded. ``nn.MultiheadAttention`` is counted explicitly
    because its Q/K/V and output projections are executed through PyTorch's
    functional path and are otherwise invisible to ordinary ``Linear`` hooks.
    """
    import torch

    total = 0
    handles = []

    def convolution_hook(module, inputs, output):
        nonlocal total
        del inputs
        batch = int(output.shape[0])
        positions = int(np.prod(output.shape[2:]))
        kernel = int(np.prod(module.kernel_size))
        per_output = kernel * module.in_channels // module.groups
        total += batch * module.out_channels * positions * per_output

    def linear_hook(module, inputs, output):
        nonlocal total
        del inputs
        total += int(output.numel()) * int(module.in_features)

    def attention_hook(module, inputs, output):
        nonlocal total
        query = inputs[0]
        if module.batch_first:
            batch, target_tokens, embedding = map(int, query.shape[:3])
        else:
            target_tokens, batch, embedding = map(int, query.shape[:3])
        key = inputs[1] if len(inputs) > 1 else query
        source_tokens = int(key.shape[1] if module.batch_first else key.shape[0])
        # Q, K, V projections plus output projection. The registered encoder
        # is self-attention, but the token dimensions are retained explicitly.
        projection = batch * (
            target_tokens * embedding * embedding
            + 2 * source_tokens * embedding * embedding
            + target_tokens * embedding * embedding
        )
        # QK^T and attention-weighted V across all heads.
        attention = 2 * batch * target_tokens * source_tokens * embedding
        total += projection + attention

    attention_children = {
        id(child)
        for module in model.modules()
        if isinstance(module, torch.nn.MultiheadAttention)
        for child in module.modules()
        if child is not module
    }

    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            handles.append(module.register_forward_hook(attention_hook))
        elif isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            handles.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, torch.nn.Linear) and id(module) not in attention_children:
            handles.append(module.register_forward_hook(linear_hook))
    mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            model(example)
    finally:
        for handle in handles:
            handle.remove()
        model.train(mode)
    return int(total)


def synchronized_batch1_latency(
    callable_model,
    example,
    *,
    device,
    warmups: int = 20,
    repeats: int = 100,
) -> dict[str, object]:
    import torch

    callable_model.eval()
    with torch.inference_mode():
        for _ in range(warmups):
            callable_model(example)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        telemetry_before = _nvidia_snapshot() if device.type == "cuda" else None
        samples = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            callable_model(example)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append(1000.0 * (perf_counter() - start))
        telemetry_after = _nvidia_snapshot() if device.type == "cuda" else None
    stability = _timing_stability(samples)
    telemetry_valid = True
    if device.type == "cuda":
        telemetry_valid = bool(
            telemetry_before
            and telemetry_after
            and max(telemetry_before["temperature_c"], telemetry_after["temperature_c"]) < 85.0
            and telemetry_before["power_limit_w"] == telemetry_after["power_limit_w"]
            and telemetry_before["pstate"] == telemetry_after["pstate"]
        )
    return {
        "median_ms": float(np.median(samples)),
        "samples_ms": samples,
        "warmups": warmups,
        "repeats": repeats,
        "batch_size": 1,
        "synchronized": True,
        "device": str(device),
        "timing_stability": stability,
        "telemetry_before": telemetry_before,
        "telemetry_after": telemetry_after,
        "telemetry_required": device.type == "cuda",
        "measurement_valid": bool(
            warmups == 20
            and repeats == 100
            and np.all(np.isfinite(samples))
            and stability["drift_valid"]
            and telemetry_valid
        ),
        "host_platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
