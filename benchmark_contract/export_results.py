from __future__ import annotations

import argparse
import subprocess

from common import discover_environment, get_result_paths, load_all_configs, read_json, write_json


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_metadata(scenario: str) -> None:
    configs = load_all_configs()
    manifest = configs["manifest"]
    experiment = configs["experiment"]["experiment"]

    payload = {
        "method_name": manifest.get("method_name"),
        "method_family": manifest.get("method_family"),
        "paper_year": manifest.get("paper_year"),
        "framework": manifest.get("framework"),
        "scenario": scenario,
        "dataset": experiment.get("dataset"),
        "deploy_file": experiment.get("deploy_file"),
        "checkpoint": experiment.get("checkpoint"),
        "commit": git_commit(),
        "notes": manifest.get("notes", []),
    }
    write_json(get_result_paths()["metadata"], payload)
    write_json(get_result_paths()["environment"], discover_environment())


def finalize() -> None:
    paths = get_result_paths()
    if not paths["metadata"].exists() or not paths["run_config"].exists():
        return

    metadata = read_json(paths["metadata"])
    run_config = read_json(paths["run_config"])
    metadata["real_inference_executed"] = run_config.get("real_inference_executed")
    metadata["reproduction_status"] = run_config.get("reproduction_status")
    metadata["backend"] = run_config.get("backend")
    write_json(paths["metadata"], metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["metadata", "finalize"], required=True)
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    if args.phase == "metadata":
        write_metadata(args.scenario)
    else:
        finalize()


if __name__ == "__main__":
    main()
