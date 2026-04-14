from __future__ import annotations

import importlib.util
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"


def load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_configs() -> Dict[str, Any]:
    return {
        "manifest": load_yaml(ROOT / "manifest.yaml"),
        "datasets": load_yaml(CONFIG_DIR / "datasets.yaml"),
        "runtime": load_yaml(CONFIG_DIR / "runtime.yaml"),
        "experiment": load_yaml(CONFIG_DIR / "experiment.yaml"),
    }


def resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _run_command(command: List[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True).strip()
    except Exception:
        return None


def discover_environment() -> Dict[str, Any]:
    caffe_spec = importlib.util.find_spec("caffe")
    configured_caffe_bin = os.environ.get("CAFFE_BIN")
    discovered_caffe_bin = configured_caffe_bin or shutil.which("caffe.bin") or shutil.which("caffe")
    env: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "contract_root": str(ROOT),
        "framework": "caffe" if caffe_spec is not None else "caffe-unavailable",
        "framework_version": None,
        "caffe_python_available": caffe_spec is not None,
        "caffe_binary": discovered_caffe_bin,
        "caffe_binary_env": configured_caffe_bin,
        "cuda_available": shutil.which("nvidia-smi") is not None,
    }

    if caffe_spec is not None:
        try:
            import caffe  # type: ignore

            env["framework_version"] = getattr(caffe, "__version__", None)
        except Exception as exc:  # pragma: no cover
            env["framework_version"] = None
            env["caffe_import_error"] = str(exc)

    driver_version = _run_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    env["driver_version"] = driver_version.splitlines()[0] if driver_version else None

    gpu_names = _run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if gpu_names:
        lines = [line.strip() for line in gpu_names.splitlines() if line.strip()]
        env["gpu_name"] = lines[0]
        env["gpu_count"] = len(lines)
    else:
        env["gpu_name"] = None
        env["gpu_count"] = 0

    return env


def get_result_paths() -> Dict[str, Path]:
    return {
        "metadata": RESULTS_DIR / "metadata.json",
        "quality": RESULTS_DIR / "quality_metrics.json",
        "efficiency": RESULTS_DIR / "efficiency_metrics.json",
        "run_config": RESULTS_DIR / "run_config.json",
        "environment": RESULTS_DIR / "environment.json",
        "predictions_npz": OUTPUTS_DIR / "predictions.npz",
    }


def latitude_band_masks(height: int, width: int, bands: List[Tuple[float, float]]) -> Dict[str, np.ndarray]:
    lat = np.linspace(-90.0, 90.0, height, endpoint=False)[:, None]
    lat = np.repeat(lat, width, axis=1)
    masks: Dict[str, np.ndarray] = {}
    for lo, hi in bands:
        key = f"{int(lo)}_{int(hi)}"
        masks[key] = (lat >= lo) & (lat < hi)
    return masks


def epe(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    err = np.linalg.norm(pred - gt, axis=0)
    if valid_mask is not None:
        valid = valid_mask.astype(bool)
        if valid.sum() == 0:
            return float("nan")
        return float(err[valid].mean())
    return float(err.mean())
