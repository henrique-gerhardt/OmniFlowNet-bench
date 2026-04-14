from __future__ import annotations

import argparse

import numpy as np

from common import epe, get_result_paths, latitude_band_masks, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    _ = parser.parse_args()

    data = np.load(get_result_paths()["predictions_npz"])
    pred = data["pred_flow"]
    gt = data["gt_flow"]
    valid_mask = data["valid_mask"].astype(bool)

    _, height, width = pred.shape
    bands = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
    masks = latitude_band_masks(height, width, bands)

    metrics = {
        "epe_global": epe(pred, gt, valid_mask),
        "epe_polar": epe(pred, gt, (masks["-90_-60"] | masks["60_90"]) & valid_mask),
        "epe_equatorial": epe(pred, gt, (masks["-30_0"] | masks["0_30"]) & valid_mask),
        "valid_pixels_ratio": float(valid_mask.mean()),
        "epe_by_latitude": {},
    }

    for key, mask in masks.items():
        metrics["epe_by_latitude"][key] = epe(pred, gt, mask & valid_mask)

    write_json(get_result_paths()["quality"], metrics)


if __name__ == "__main__":
    main()
