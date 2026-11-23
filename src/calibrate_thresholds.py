"""
Calibrate the risk-tier thresholds on the VALIDATION split (never the test split).

The default thresholds in src/uncertainty.py were a starting guess. This script picks
them from data so that each tier means something:

  Moderate  = lowest mean-probability cut-off whose pixels burn at least 15% of the time
  High      = ... at least 40% of the time
  Extreme   = ... at least 60% of the time (needs >= 300 supporting pixels)
  Uncertainty bump = 90th percentile of MC-Dropout std among pixels at/above the Moderate cut-off,
                     i.e. escalate the ~10% most uncertain candidate pixels one tier.

Usage (repo root):
    python -m src.calibrate_thresholds --data_dir DATA_DIR --light_ckpt checkpoints/lightweight_best.pt
Writes thresholds.json, which src/export_demo.py reads via --thresholds.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import get_dataloaders
from src.training.train import build_model
from src.uncertainty import mc_dropout_predict

TARGETS = {"prob_moderate": 0.15, "prob_high": 0.40, "prob_extreme": 0.60}
MIN_SUPPORT = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--light_ckpt", default="checkpoints/lightweight_best.pt")
    ap.add_argument("--out", default="thresholds.json")
    ap.add_argument("--mc_samples", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, _ = get_dataloaders(args.data_dir, batch_size=32)
    x = val_loader.dataset.inputs
    y = val_loader.dataset.labels[:, 0] >= 0.5

    model = build_model("lightweight").to(device)
    model.load_state_dict(torch.load(args.light_ckpt, map_location=device)["model_state"])

    means, stds = [], []
    for i in range(0, len(x), 64):
        xb = torch.from_numpy(x[i:i + 64]).float().to(device)
        m, s = mc_dropout_predict(model, xb, n_samples=args.mc_samples)
        means.append(m.cpu().numpy()[:, 0])
        stds.append(s.cpu().numpy()[:, 0])
    mean = np.concatenate(means).reshape(-1)
    std = np.concatenate(stds).reshape(-1)
    yy = y.reshape(-1)
    print(f"validation pixels: {mean.size}, fire rate {yy.mean():.4%}")

    # cumulative precision at each candidate cut-off
    cuts = np.round(np.arange(0.05, 0.96, 0.01), 2)
    prec, support = [], []
    for c in cuts:
        sel = mean >= c
        n = int(sel.sum())
        support.append(n)
        prec.append(float(yy[sel].mean()) if n else 0.0)
    prec, support = np.array(prec), np.array(support)

    found = {}
    for name, target in TARGETS.items():
        ok = np.where((prec >= target) & (support >= MIN_SUPPORT))[0]
        found[name] = float(cuts[ok[0]]) if len(ok) else None
    # fall back to the highest cut-off that still has enough support
    max_supported = float(cuts[np.where(support >= MIN_SUPPORT)[0][-1]])
    for name in TARGETS:
        if found[name] is None:
            found[name] = max_supported
    # enforce strictly increasing cut-offs with at least 0.05 gaps
    found["prob_high"] = max(found["prob_high"], round(found["prob_moderate"] + 0.05, 2))
    found["prob_extreme"] = max(found["prob_extreme"], round(found["prob_high"] + 0.05, 2))

    cand = mean >= found["prob_moderate"]
    bump = float(np.percentile(std[cand], 90)) if cand.sum() else 0.05

    out = {**{k: round(v, 3) for k, v in found.items()}, "uncertainty_bump": round(bump, 4),
           "method": "validation-split calibration: cumulative precision targets "
                     f"{TARGETS}, min support {MIN_SUPPORT} px; bump = p90 of std among candidate pixels",
           "validation_pixels": int(mean.size)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("cumulative precision at chosen cut-offs:",
          {k: round(float(prec[int(np.argmin(np.abs(cuts - found[k])))]), 3) for k in TARGETS})


if __name__ == "__main__":
    main()
