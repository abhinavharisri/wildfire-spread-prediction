"""
Creates a SYNTHETIC demo/ folder (fake fire maps, placeholder metrics) purely so the
Streamlit app and HTML dashboard can be previewed/tested before real training results
exist. Everything it writes is flagged meta.synthetic = true and the dashboard shows a
"SAMPLE DATA" banner. Never present this as a result.

    python tools/make_sample_demo.py --out demo_sample
"""
import argparse
import json
import os

import numpy as np


def blob(rng, cx, cy, sx, sy, n=64):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))


def smooth(a, k=3):
    for _ in range(k):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_sample")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(1)
    N = args.n
    x = rng.normal(size=(N, 12, 64, 64)).astype(np.float32) * 0.3
    y = np.zeros((N, 64, 64), np.uint8)
    att = np.zeros((N, 64, 64), np.float32)
    mc_mean = np.zeros_like(att)
    mc_std = np.zeros_like(att)
    for i in range(N):
        cx, cy = rng.uniform(16, 48, 2)
        sx, sy = rng.uniform(2.5, 7, 2)
        prev = (blob(rng, cx, cy, sx, sy) > 0.5).astype(np.float32)
        dx, dy = rng.normal(0, 3, 2)
        nxt = blob(rng, cx + dx, cy + dy, sx * 1.5 + 1, sy * 1.5 + 1)
        lab = (nxt > 0.45).astype(np.uint8)
        if i % 10 == 9:
            prev[:] = 0
            lab[:] = 0
        p = prev.mean()
        x[i, 11] = (prev - p) / (prev.std() + 1e-6) if prev.std() > 0 else 0
        y[i] = lab
        base = smooth(nxt, 2)
        att[i] = np.clip(base * 0.9 + rng.normal(0, 0.06, (64, 64)), 0, 1)
        mc_mean[i] = np.clip(smooth(nxt, 1) * 1.0 + rng.normal(0, 0.05, (64, 64)), 0, 1)
        edge = 4 * mc_mean[i] * (1 - mc_mean[i])
        mc_std[i] = np.clip(0.02 + 0.22 * edge + rng.normal(0, 0.01, (64, 64)), 0, 0.5)

    fire = y.reshape(N, -1).mean(1)
    order = np.argsort(fire)
    featured = order[np.linspace(0, N - 1, min(24, N)).astype(int)]
    np.savez_compressed(os.path.join(args.out, "demo_bundle.npz"),
                        x=x.astype(np.float16), y=y, att=att.astype(np.float16),
                        mc_mean=mc_mean.astype(np.float16), mc_std=mc_std.astype(np.float16),
                        orig_idx=np.arange(N), featured=featured)

    rec = np.linspace(0, 1, 200)
    metrics = {
        "meta": {"synthetic": True, "n_test_patches": 1689, "fire_pixel_fraction": 0.0108,
                 "mc_samples": 20, "tiers": ["Low", "Moderate", "High", "Extreme"],
                 "default_thresholds": {"prob_moderate": 0.3, "prob_high": 0.6,
                                        "prob_extreme": 0.85, "uncertainty_bump": 0.15}},
        "attention": {"accuracy": {"AUC-PR": 0.2267, "F1": 0.3286, "IoU": 0.1966},
                      "efficiency": {"parameters": 7854365, "parameters_M": 7.854, "FLOPs_G": 0.8876,
                                     "latency_ms_per_image": 4.026, "device": "cuda"},
                      "efficiency_cpu": {"latency_ms_per_image": 38.0, "device": "cpu"}},
        "lightweight": {"accuracy": {"AUC-PR": 0.3114, "F1": 0.383, "IoU": 0.2368},
                        "efficiency": {"parameters": 1447957, "parameters_M": 1.448, "FLOPs_G": 0.1416,
                                       "latency_ms_per_image": 3.804, "device": "cuda"},
                        "efficiency_cpu": {"latency_ms_per_image": 9.0, "device": "cpu"}},
        "lightweight_mc": {"accuracy": {"AUC-PR": 0.31, "F1": 0.38, "IoU": 0.235}},
        "pr_curves": {"attention": {"recall": rec.round(4).tolist(),
                                    "precision": (0.9 * np.exp(-3.2 * rec) + 0.01).round(4).tolist()},
                      "lightweight": {"recall": rec.round(4).tolist(),
                                      "precision": (0.95 * np.exp(-2.8 * rec) + 0.01).round(4).tolist()}},
        "f1_sweep": {"thresholds": [round(0.05 * k, 2) for k in range(1, 20)],
                     "attention": [round(0.33 - 0.5 * (0.05 * k - 0.5) ** 2, 4) for k in range(1, 20)],
                     "lightweight": [round(0.38 - 0.5 * (0.05 * k - 0.55) ** 2, 4) for k in range(1, 20)]},
        "risk_calibration": [
            {"tier": "Low", "pixels": 6800000, "pct_of_pixels": 96.5, "observed_fire_rate": 0.004},
            {"tier": "Moderate", "pixels": 150000, "pct_of_pixels": 2.1, "observed_fire_rate": 0.21},
            {"tier": "High", "pixels": 70000, "pct_of_pixels": 1.0, "observed_fire_rate": 0.42},
            {"tier": "Extreme", "pixels": 30000, "pct_of_pixels": 0.4, "observed_fire_rate": 0.66}],
        "uncertainty_stats": {"mean_std_correct_pixels": 0.021, "mean_std_wrong_pixels": 0.087,
                              "fraction_pixels_bumped": 0.018},
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        ep = list(range(1, 19))
        json.dump({"attention": {"epoch": ep, "train_loss": [0.0071 - 0.0001 * e for e in ep],
                                 "val_loss": [0.0080 - 0.0002 * min(e, 10) + 0.0001 * max(0, e - 10) for e in ep]},
                   "lightweight": {"epoch": ep[:17], "train_loss": [0.0058 - 0.0001 * e for e in ep[:17]],
                                   "val_loss": [0.0072 - 0.0001 * min(e, 11) for e in ep[:17]]}}, f)
    print("wrote synthetic demo to", args.out)


if __name__ == "__main__":
    main()
