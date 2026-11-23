"""
Backend walkthrough: runs the real pipeline on one test patch and prints every stage.

    python tools/backend_demo.py                # picks the patch with the most fire
    python tools/backend_demo.py --patch 12     # a specific patch in the demo bundle
    python tools/backend_demo.py --pause        # wait for Enter between steps (for the panel)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.lightweight_unet import LightweightUNet
from src.uncertainty import RISK_TIERS, RiskThresholds, assign_risk_tier, mc_dropout_predict, risk_tier_report

CHANNELS = ["Elevation", "Wind direction", "Wind speed", "Min temp", "Max temp", "Humidity",
            "Precipitation", "Drought index", "NDVI", "Population", "Energy release", "Fire today"]
STAGES = ["enc1", "enc2", "enc3", "enc4", "bottleneck", "bottleneck_attn",
          "dec4", "dec3", "dec2", "dec1", "out_conv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default=os.environ.get("WILDFIRE_DEMO_DIR", "demo"))
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--pause", action="store_true")
    a = ap.parse_args()

    def step(title):
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        if a.pause:
            input("(press Enter) ")

    step("STEP 1  Load trained model weights")
    meta = json.load(open(os.path.join(a.demo, "metrics.json")))
    t = time.perf_counter()
    model = LightweightUNet(in_channels=12, base_ch=24, dropout_p=0.2)
    model.load_state_dict(torch.load(os.path.join(a.demo, "lightweight_best.pt"), map_location="cpu")["model_state"])
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Lightweight U-Net loaded in {1000 * (time.perf_counter() - t):.0f} ms | parameters: {n_par:,}")

    step("STEP 2  Take one test patch (input from the NDWS dataset)")
    z = np.load(os.path.join(a.demo, "demo_bundle.npz"))
    fire = z["y"].reshape(len(z["y"]), -1).mean(1)
    i = a.patch if a.patch is not None else int(np.argmax(fire))
    x = z["x"][i].astype(np.float32)
    y = z["y"][i]
    print(f"Patch #{int(z['orig_idx'][i])} | input tensor shape {x.shape} (channels, height, width)")
    for k, nm in enumerate(CHANNELS):
        print(f"  ch{k:02d} {nm:<14} min {x[k].min():7.2f}  max {x[k].max():7.2f}  mean {x[k].mean():7.2f}")
    print(f"Ground-truth fire pixels tomorrow: {int(y.sum())} of {y.size} ({100 * y.mean():.2f}%)")

    step("STEP 3  Forward pass: shape of the data after every stage")
    xt = torch.from_numpy(x[None])
    rows, hs = [], []
    t0 = [0.0]
    for nm in STAGES:
        if hasattr(model, nm):
            def hk(_m, _i, out, nm=nm):
                o = out[0] if isinstance(out, (tuple, list)) else out
                rows.append((nm, tuple(o.shape), 1000 * (time.perf_counter() - t0[0])))
            hs.append(getattr(model, nm).register_forward_hook(hk))
    with torch.no_grad():
        t0[0] = time.perf_counter()
        p = model(xt)
    for h in hs:
        h.remove()
    print(f"{'stage':<18}{'output shape':<26}{'cumulative ms':>14}")
    for nm, sh, ms in rows:
        print(f"{nm:<18}{str(sh):<26}{ms:>14.1f}")
    print(f"sigmoid output: {tuple(p.shape)}  range [{p.min().item():.3f}, {p.max().item():.3f}]")

    step(f"STEP 4  Monte-Carlo Dropout: {a.samples} stochastic passes -> mean and uncertainty")
    t = time.perf_counter()
    mean, std = mc_dropout_predict(model, xt, n_samples=a.samples)
    ms = 1000 * (time.perf_counter() - t)
    mean, std = mean.numpy()[0, 0], std.numpy()[0, 0]
    print(f"{a.samples} passes in {ms:.0f} ms ({ms / a.samples:.1f} ms each)")
    print(f"mean probability: min {mean.min():.3f} max {mean.max():.3f}")
    print(f"uncertainty (std): min {std.min():.4f} max {std.max():.4f} mean {std.mean():.4f}")
    print(f"pixels predicted fire (mean >= 0.5): {int((mean >= 0.5).sum())}  | actual: {int(y.sum())}")

    step("STEP 5  Convert to risk tiers (thresholds calibrated on the validation split)")
    d = meta["meta"]["default_thresholds"]
    thr = RiskThresholds(d["prob_moderate"], d["prob_high"], d["prob_extreme"], d["uncertainty_bump"])
    print(f"thresholds: {thr}")
    tiers = assign_risk_tier(mean, std, thr)
    for k, nm in enumerate(RISK_TIERS):
        n = int((tiers == k).sum())
        print(f"  {nm:<9}{n:>6} pixels  ({100 * n / tiers.size:5.2f}%)")
    print("\nRecommendation report:")
    print(json.dumps(risk_tier_report(mean, std, thr), indent=2, default=str))

    step("DONE  Same code path the Streamlit app and the evaluation use")
    print("Stages: load -> normalised input -> U-Net forward -> MC-Dropout -> risk tiers -> recommendation")


if __name__ == "__main__":
    main()
