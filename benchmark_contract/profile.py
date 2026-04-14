from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import get_result_paths, load_all_configs, write_json
from run_inference import load_method_model, load_sample_pair, predict_flow, resolve_checkpoint_path


def count_parameters(model: Dict[str, Any]) -> int | None:
    if model.get("backend") != "pycaffe":
        return None

    total = 0
    for blobs in model["net"].params.values():
        for blob in blobs:
            total += int(np.prod(blob.data.shape))
    return total


def checkpoint_size_mb(configs: Dict[str, Any]) -> float | None:
    checkpoint = resolve_checkpoint_path(configs)
    if checkpoint is None or not checkpoint.exists():
        return None
    return round(checkpoint.stat().st_size / (1024 * 1024), 4)


def query_gpu_memory_mb() -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    except Exception:
        return None

    values = [line.strip() for line in output.splitlines() if line.strip()]
    if not values:
        return None
    try:
        return float(values[0])
    except ValueError:
        return None


def measure_latency(model: Dict[str, Any], sample: Dict[str, Any], warmup: int, runs: int) -> Dict[str, Any]:
    if model.get("backend") not in {"pycaffe", "caffe-cli"}:
        return {
            "latency_mean_ms": None,
            "latency_median_ms": None,
            "latency_p95_ms": None,
            "max_gpu_memory_mb": None,
            "fps": None,
            "notes": [
                "Latency and GPU memory were not measured because real CAFFE inference is unavailable in this workspace."
            ],
        }

    times_ms: List[float] = []
    sampled_memory_mb: List[float] = []

    for _ in range(warmup):
        predict_flow(model, sample)

    for _ in range(runs):
        start = time.perf_counter()
        predict_flow(model, sample)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
        memory_mb = query_gpu_memory_mb()
        if memory_mb is not None:
            sampled_memory_mb.append(memory_mb)

    times_sorted = sorted(times_ms)
    p95_index = max(0, int(np.ceil(0.95 * len(times_sorted))) - 1)
    mean_ms = float(statistics.mean(times_ms)) if times_ms else None

    notes: List[str] = []
    if sampled_memory_mb:
        notes.append("GPU memory is the maximum sampled value from nvidia-smi after each inference, not a hardware peak trace.")

    return {
        "latency_mean_ms": mean_ms,
        "latency_median_ms": float(statistics.median(times_ms)) if times_ms else None,
        "latency_p95_ms": float(times_sorted[p95_index]) if times_sorted else None,
        "max_gpu_memory_mb": max(sampled_memory_mb) if sampled_memory_mb else None,
        "fps": float(1000.0 / mean_ms) if mean_ms and mean_ms > 0 else None,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    _ = parser.parse_args()

    configs = load_all_configs()
    runtime = configs["runtime"]["runtime"]
    sample = load_sample_pair(configs)
    model = load_method_model(configs, sample)

    metrics = {
        "parameters": count_parameters(model),
        "checkpoint_size_mb": checkpoint_size_mb(configs),
        "flops_g": None,
        "notes": [],
    }
    metrics["notes"].extend(model.get("notes", []))
    metrics["notes"].append(
        "FLOP counting remains null because the workspace does not expose a trustworthy analyzer for the custom CAFFE graph."
    )

    latency = measure_latency(
        model,
        sample,
        warmup=int(runtime["warmup_runs"]),
        runs=int(runtime["measured_runs"]),
    )
    metrics.update({key: value for key, value in latency.items() if key != "notes"})
    metrics["notes"].extend(latency.get("notes", []))

    write_json(get_result_paths()["efficiency"], metrics)


if __name__ == "__main__":
    main()
