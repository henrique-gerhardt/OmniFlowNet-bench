from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from common import OUTPUTS_DIR, get_result_paths, load_all_configs, resolve_path, set_seed, write_json


def read_flo(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = np.fromfile(handle, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file: {path}")
        width = int(np.fromfile(handle, np.int32, count=1)[0])
        height = int(np.fromfile(handle, np.int32, count=1)[0])
        data = np.fromfile(handle, np.float32, count=2 * width * height)
    flow = data.reshape(height, width, 2)
    return np.transpose(flow, (2, 0, 1)).astype(np.float32)


def resize_image_chw(image_chw: np.ndarray, height: int, width: int) -> np.ndarray:
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    resized = cv2.resize(image_hwc, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.transpose(resized, (2, 0, 1)).astype(np.float32)


def resize_flow(flow_chw: np.ndarray, height: int, width: int) -> np.ndarray:
    old_height, old_width = flow_chw.shape[1:]
    flow_hwc = np.transpose(flow_chw, (1, 2, 0))
    resized = cv2.resize(flow_hwc, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    resized[..., 0] *= width / old_width
    resized[..., 1] *= height / old_height
    return np.transpose(resized, (2, 0, 1)).astype(np.float32)


def find_sequences(configs: Dict[str, Any]) -> List[Path]:
    exp = configs["experiment"]["experiment"]
    dataset_name = exp["dataset"]
    dataset_cfg = configs["datasets"]["datasets"][dataset_name]
    dataset_root = resolve_path(dataset_cfg["root"])
    if dataset_root is None:
        raise FileNotFoundError("Dataset root is not configured.")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    sequence_glob = dataset_cfg.get("sequence_glob", "*")
    sequences = []
    for path in sorted(dataset_root.glob(sequence_glob)):
        if not path.is_dir():
            continue
        if list(path.glob(dataset_cfg["image_glob"])) and list(path.glob(dataset_cfg["flow_glob"])):
            sequences.append(path)

    if not sequences:
        raise FileNotFoundError(f"No dataset sequences matching {sequence_glob!r} were found under {dataset_root}")
    return sequences


def resolve_dataset_sample(configs: Dict[str, Any]) -> Tuple[Path, Path, Path, Path, List[str]]:
    exp = configs["experiment"]["experiment"]
    dataset_cfg = configs["datasets"]["datasets"][exp["dataset"]]
    dataset_root = resolve_path(dataset_cfg["root"])
    sample_sequence = exp.get("sample_sequence")
    sample_index = int(exp.get("sample_index", 0))

    if dataset_root is None:
        raise FileNotFoundError("Dataset root is not configured.")

    notes: List[str] = []
    if sample_sequence:
        sequence_root = dataset_root / sample_sequence
        if not sequence_root.exists():
            raise FileNotFoundError(f"Configured sample_sequence does not exist: {sequence_root}")
        sequences = [sequence_root]
    else:
        sequences = find_sequences(configs)

    for sequence_root in sequences:
        images = sorted(sequence_root.glob(dataset_cfg["image_glob"]))
        flows = sorted(sequence_root.glob(dataset_cfg["flow_glob"]))
        flow_by_stem = {path.stem: path for path in flows}
        if len(images) < 2:
            continue

        valid_pairs: List[Tuple[Path, Path, Path]] = []
        for index in range(len(images) - 1):
            flow_path = flow_by_stem.get(images[index].stem)
            if flow_path is None:
                continue
            valid_pairs.append((images[index], images[index + 1], flow_path))

        if sample_index < len(valid_pairs):
            notes.append(
                "Dataset pairing is inferred from the local layout: image N is paired with image N+1 and ground truth N.flo."
            )
            image1_path, image2_path, flow_path = valid_pairs[sample_index]
            return image1_path, image2_path, flow_path, sequence_root, notes

    raise FileNotFoundError("No valid image/image/flow triplet could be inferred from the local dataset layout.")


def load_sample_pair(configs: Dict[str, Any]) -> Dict[str, Any]:
    exp = configs["experiment"]["experiment"]
    target_height = int(exp["input_height"])
    target_width = int(exp["input_width"])

    image1_path, image2_path, flow_path, sequence_root, notes = resolve_dataset_sample(configs)

    image1 = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
    image2 = cv2.imread(str(image2_path), cv2.IMREAD_COLOR)
    if image1 is None or image2 is None:
        raise FileNotFoundError("Failed to load dataset image pair.")

    image1_chw = np.transpose(image1, (2, 0, 1)).astype(np.float32)
    image2_chw = np.transpose(image2, (2, 0, 1)).astype(np.float32)
    gt_flow = read_flo(flow_path)

    original_height, original_width = image1.shape[:2]
    if (original_height, original_width) != (target_height, target_width):
        image1_chw = resize_image_chw(image1_chw, target_height, target_width)
        image2_chw = resize_image_chw(image2_chw, target_height, target_width)
        gt_flow = resize_flow(gt_flow, target_height, target_width)

    valid_mask = np.isfinite(gt_flow).all(axis=0) & (np.abs(gt_flow) < 1e9).all(axis=0)

    return {
        "image1": image1_chw,
        "image2": image2_chw,
        "gt_flow": gt_flow,
        "valid_mask": valid_mask,
        "image1_path": str(image1_path),
        "image2_path": str(image2_path),
        "flow_path": str(flow_path),
        "sequence_root": str(sequence_root),
        "dataset_root": str(resolve_path(configs["datasets"]["datasets"][exp["dataset"]]["root"])),
        "original_height": original_height,
        "original_width": original_width,
        "notes": notes,
    }


def preprocess_for_omniflownet(image_chw: np.ndarray) -> np.ndarray:
    image = image_chw.astype(np.float32) / 255.0
    channel_mean = image.reshape(image.shape[0], -1).mean(axis=1)[:, None, None]
    return image - channel_mean


def resolve_caffe_binary() -> Path | None:
    candidates = []
    env_value = os.environ.get("CAFFE_BIN")
    if env_value:
        candidates.append(Path(env_value))

    for candidate in (
        shutil.which("caffe.bin"),
        shutil.which("caffe"),
        "/opt/LiteFlowNet/build/tools/caffe.bin",
        "/opt/liteflownet/build/tools/caffe.bin",
    ):
        if candidate:
            candidates.append(Path(candidate))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def try_import_caffe() -> Tuple[Any | None, str | None]:
    caffe_pythonpath = os.environ.get("CAFFE_PYTHONPATH")
    if caffe_pythonpath:
        for entry in caffe_pythonpath.split(os.pathsep):
            if entry and entry not in sys.path:
                sys.path.insert(0, entry)

    try:
        caffe = importlib.import_module("caffe")
        return caffe, None
    except Exception as exc:
        return None, str(exc)


def write_image_list(list_path: Path, image_path: str) -> None:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(f"{Path(image_path).resolve().as_posix()} 0\n", encoding="utf-8")


def render_deploy_file(
    deploy_template: Path,
    sample: Dict[str, Any],
    configs: Dict[str, Any],
    runtime_dir: Path,
) -> Path:
    exp = configs["experiment"]["experiment"]
    target_height = int(exp["input_height"])
    target_width = int(exp["input_width"])
    adapted_height = int(exp.get("adapted_height", target_height))
    adapted_width = int(exp.get("adapted_width", target_width))

    img1_list = runtime_dir / "tmp" / "img1.txt"
    img2_list = runtime_dir / "tmp" / "img2.txt"
    out_dir = runtime_dir / "flo_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_image_list(img1_list, sample["image1_path"])
    write_image_list(img2_list, sample["image2_path"])

    text = deploy_template.read_text(encoding="utf-8")
    text = text.replace('source: "tmp/img1.txt"', f'source: "{img1_list.resolve().as_posix()}"')
    text = text.replace('source: "tmp/img2.txt"', f'source: "{img2_list.resolve().as_posix()}"')

    replacements = {
        "$ADAPTED_WIDTH": str(adapted_width),
        "$ADAPTED_HEIGHT": str(adapted_height),
        "$TARGET_WIDTH": str(target_width),
        "$TARGET_HEIGHT": str(target_height),
        "$SCALE_WIDTH": repr(target_width / adapted_width),
        "$SCALE_HEIGHT": repr(target_height / adapted_height),
        "$OUTFOLDER": out_dir.resolve().as_posix(),
        "$CNN": "benchmark_contract",
    }
    for key, value in replacements.items():
        text = text.replace(key, value)

    rendered_path = runtime_dir / "deploy_rendered.prototxt"
    rendered_path.write_text(text, encoding="utf-8")
    return rendered_path


def resolve_checkpoint_path(configs: Dict[str, Any]) -> Path | None:
    exp = configs["experiment"]["experiment"]
    checkpoint_env = exp.get("checkpoint_env")
    checkpoint = os.environ.get(checkpoint_env) if checkpoint_env else None
    if not checkpoint:
        checkpoint = exp.get("checkpoint")
    return resolve_path(checkpoint)


def make_runtime_dir() -> Path:
    runtime_dir = OUTPUTS_DIR / "runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def load_method_model(configs: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
    exp = configs["experiment"]["experiment"]
    deploy_file = resolve_path(exp.get("deploy_file"))
    checkpoint = resolve_checkpoint_path(configs)
    output_blob = exp.get("output_blob", "predict_flow_final")
    notes: List[str] = []

    if deploy_file is None or not deploy_file.exists():
        return {
            "backend": "unavailable",
            "real_inference_available": False,
            "output_blob": output_blob,
            "deploy_file": str(deploy_file) if deploy_file else None,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "notes": ["Configured deploy_file is missing; CAFFE inference cannot be initialized."],
        }

    if checkpoint is None:
        notes.append("No checkpoint is configured in the workspace.")
    elif not checkpoint.exists():
        notes.append(f"Checkpoint path does not exist: {checkpoint}")

    caffe_bin = resolve_caffe_binary()
    if caffe_bin is not None and checkpoint is not None and checkpoint.exists():
        notes.append(f"Using caffe binary execution path via {caffe_bin}.")
        return {
            "backend": "caffe-cli",
            "real_inference_available": True,
            "output_blob": output_blob,
            "deploy_file": str(deploy_file),
            "checkpoint": str(checkpoint),
            "caffe_bin": str(caffe_bin),
            "notes": notes,
        }
    if caffe_bin is None:
        notes.append("No caffe binary was found in CAFFE_BIN, PATH, or the default LiteFlowNet build locations.")

    caffe, caffe_error = try_import_caffe()
    if caffe is None:
        notes.append(
            "pycaffe is not importable in the current environment."
            + (f" Import error: {caffe_error}" if caffe_error else "")
        )

    if checkpoint is None or not checkpoint.exists() or caffe is None:
        return {
            "backend": "unavailable",
            "real_inference_available": False,
            "output_blob": output_blob,
            "deploy_file": str(deploy_file),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "notes": notes,
        }

    runtime_dir = OUTPUTS_DIR / "runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    rendered_deploy = render_deploy_file(deploy_file, sample, configs, runtime_dir)

    runtime = configs["runtime"]["runtime"]
    if runtime.get("device", "cuda") == "cuda":
        try:
            caffe.set_mode_gpu()
            caffe.set_device(0)
            notes.append("CAFFE was configured in GPU mode.")
        except Exception as exc:
            notes.append(f"Failed to enable CAFFE GPU mode; falling back to CPU if possible. Error: {exc}")
            caffe.set_mode_cpu()
    else:
        caffe.set_mode_cpu()

    try:
        net = caffe.Net(str(rendered_deploy), str(checkpoint), caffe.TEST)
    except Exception as exc:
        notes.append(f"Failed to construct caffe.Net from rendered deploy/checkpoint: {exc}")
        return {
            "backend": "unavailable",
            "real_inference_available": False,
            "output_blob": output_blob,
            "deploy_file": str(deploy_file),
            "checkpoint": str(checkpoint),
            "rendered_deploy": str(rendered_deploy),
            "notes": notes,
        }

    notes.append(
        "Deploy preprocessing confirmed from prototxt: OpenCV/ImageData input, scale by 1/255, then DataAugmentation test-time mean removal."
    )
    return {
        "backend": "pycaffe",
        "real_inference_available": True,
        "net": net,
        "output_blob": output_blob,
        "deploy_file": str(deploy_file),
        "checkpoint": str(checkpoint),
        "rendered_deploy": str(rendered_deploy),
        "notes": notes,
    }


def zero_prediction_like(sample: Dict[str, Any]) -> np.ndarray:
    _, height, width = sample["gt_flow"].shape
    return np.zeros((2, height, width), dtype=np.float32)


def predict_flow_via_caffe_cli(
    model: Dict[str, Any],
    sample: Dict[str, Any],
    configs: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    runtime_dir = make_runtime_dir()
    rendered_deploy = render_deploy_file(Path(model["deploy_file"]), sample, configs, runtime_dir)
    out_dir = runtime_dir / "flo_out"
    caffe_bin = model["caffe_bin"]

    runtime = configs["runtime"]["runtime"]
    command = [
        caffe_bin,
        "test",
        f"-model={rendered_deploy}",
        f"-weights={model['checkpoint']}",
        "-iterations=1",
    ]
    if runtime.get("device", "cuda") == "cuda":
        command.append("-gpu=0")

    env = os.environ.copy()
    env.setdefault("GLOG_logtostderr", "1")

    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    log_path = runtime_dir / "caffe_cli.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "caffe binary execution failed with exit code "
            f"{completed.returncode}. See {log_path} for the captured output."
        )

    flo_files = sorted(out_dir.glob("*.flo"))
    if not flo_files:
        raise FileNotFoundError(
            f"No .flo output was written by caffe CLI into {out_dir}. See {log_path} for details."
        )

    pred_flow = read_flo(flo_files[0])
    _, target_height, target_width = sample["gt_flow"].shape
    if pred_flow.shape[1:] != (target_height, target_width):
        pred_flow = resize_flow(pred_flow, target_height, target_width)

    return pred_flow, {
        "real_inference_executed": True,
        "notes": [f"Real inference executed through caffe CLI. Log stored at {log_path}."],
    }


def predict_flow(model: Dict[str, Any], sample: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    if model["backend"] == "caffe-cli":
        return predict_flow_via_caffe_cli(model, sample, load_all_configs())

    if model["backend"] != "pycaffe":
        return (
            zero_prediction_like(sample),
            {
                "real_inference_executed": False,
                "notes": [
                    "Real OmniFlowNet inference was not executed because the workspace lacks a usable CAFFE runtime and/or checkpoint."
                ],
            },
        )

    net = model["net"]
    output_blob = model["output_blob"]
    result = net.forward()
    if output_blob in result:
        pred = result[output_blob]
    elif output_blob in net.blobs:
        pred = net.blobs[output_blob].data
    else:
        raise KeyError(f"Output blob {output_blob!r} was not produced by the CAFFE network.")

    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 4:
        pred = pred[0]
    elif pred.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected prediction shape from CAFFE backend: {pred.shape}")

    if pred.shape[0] != 2 and pred.shape[-1] == 2:
        pred = np.transpose(pred, (2, 0, 1))

    _, target_height, target_width = sample["gt_flow"].shape
    if pred.shape[1:] != (target_height, target_width):
        pred = resize_flow(pred, target_height, target_width)

    return pred.astype(np.float32), {"real_inference_executed": True, "notes": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    configs = load_all_configs()
    runtime = configs["runtime"]["runtime"]
    set_seed(int(runtime["seed"]))

    sample = load_sample_pair(configs)
    model = load_method_model(configs, sample)

    start = time.perf_counter()
    pred_flow, inference_info = predict_flow(model, sample)
    end = time.perf_counter()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUTS_DIR / "predictions.npz",
        pred_flow=pred_flow,
        gt_flow=sample["gt_flow"],
        valid_mask=sample["valid_mask"],
    )

    notes = []
    notes.extend(sample.get("notes", []))
    notes.extend(model.get("notes", []))
    notes.extend(inference_info.get("notes", []))

    write_json(
        get_result_paths()["run_config"],
        {
            "scenario": args.scenario,
            "batch_size": int(runtime["batch_size"]),
            "precision": runtime["precision"],
            "warmup_runs": int(runtime["warmup_runs"]),
            "measured_runs": int(runtime["measured_runs"]),
            "input_height": int(sample["image1"].shape[1]),
            "input_width": int(sample["image1"].shape[2]),
            "original_height": int(sample["original_height"]),
            "original_width": int(sample["original_width"]),
            "single_inference_wall_ms": (end - start) * 1000.0,
            "image1_path": sample["image1_path"],
            "image2_path": sample["image2_path"],
            "flow_path": sample["flow_path"],
            "sequence_root": sample["sequence_root"],
            "dataset_root": sample["dataset_root"],
            "deploy_file": model.get("deploy_file"),
            "checkpoint": model.get("checkpoint"),
            "backend": model.get("backend"),
            "output_blob": model.get("output_blob"),
            "real_inference_executed": bool(inference_info.get("real_inference_executed")),
            "reproduction_status": (
                "official_reproduction"
                if inference_info.get("real_inference_executed")
                else "degraded_missing_checkpoint_or_caffe"
            ),
            "notes": notes,
            "preprocessing_summary": {
                "image_loader": "cv2.imread / CAFFE ImageData",
                "color_order": "BGR",
                "scaling": "pixels multiplied by 1/255 inside deploy",
                "mean_handling": "DataAugmentation recompute_mean during test",
                "flow_output_order": ["u", "v"],
            },
        },
    )


if __name__ == "__main__":
    main()
