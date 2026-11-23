"""
Demo export
===========
Run ONCE on a GPU box (Kaggle/Colab) after both models are trained. It produces
everything the Streamlit app and the HTML dashboard need, so neither needs the
2 GB dataset or a GPU on review day:

    demo/metrics.json       accuracy + efficiency (GPU and CPU), PR curves, F1 sweep,
                            risk-tier calibration, uncertainty stats
    demo/demo_bundle.npz    N test patches: normalized inputs, ground truth,
                            attention prediction, MC-Dropout mean/std
    demo/attention_best.pt, demo/lightweight_best.pt   (copied checkpoints)

Usage (from repo root):
    python -m src.export_demo --data_dir data/raw \
        --attn_ckpt checkpoints/attention_best.pt \
        --light_ckpt checkpoints/lightweight_best.pt
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import get_dataloaders
from src.evaluation.evaluate import accuracy_metrics, efficiency_metrics
from src.training.train import build_model
from src.uncertainty import RISK_TIERS, RiskThresholds, assign_risk_tier, mc_dropout_predict


def load_model(name, ckpt, device):
    model = build_model(name).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_all(model, x_np, device, bs=64):
    """Deterministic predictions (dropout off) -> (N, 64, 64) float32."""
    model.eval()
    out = []
    for i in range(0, len(x_np), bs):
        xb = torch.from_numpy(x_np[i:i + bs]).float().to(device)
        out.append(model(xb).cpu().numpy()[:, 0])
    return np.concatenate(out)


def mc_predict_all(model, x_np, device, n_samples, bs=64):
    """MC-Dropout mean/std -> two (N, 64, 64) float32 arrays."""
    means, stds = [], []
    for i in range(0, len(x_np), bs):
        xb = torch.from_numpy(x_np[i:i + bs]).float().to(device)
        m, s = mc_dropout_predict(model, xb, n_samples=n_samples)
        means.append(m.cpu().numpy()[:, 0])
        stds.append(s.cpu().numpy()[:, 0])
    return np.concatenate(means), np.concatenate(stds)


def f1_sweep(pred, true, thresholds):
    t = true.astype(bool).reshape(-1)
    p = pred.reshape(-1)
    out = []
    for th in thresholds:
        b = p >= th
        tp = np.logical_and(b, t).sum()
        fp = np.logical_and(b, ~t).sum()
        fn = np.logical_and(~b, t).sum()
        out.append(float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0)
    return out


def pr_curve(pred, true, n_points=200):
    prec, rec, _ = precision_recall_curve(true.reshape(-1).astype(np.uint8), pred.reshape(-1))
    idx = np.linspace(0, len(prec) - 1, n_points).astype(int)
    return {"precision": [round(float(v), 4) for v in prec[idx]],
            "recall": [round(float(v), 4) for v in rec[idx]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--attn_ckpt", default="checkpoints/attention_best.pt")
    ap.add_argument("--light_ckpt", default="checkpoints/lightweight_best.pt")
    ap.add_argument("--out_dir", default="demo")
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--n_patches", type=int, default=200, help="test patches saved for the live demo")
    ap.add_argument("--thresholds", default=None, help="thresholds.json from calibrate_thresholds")
    ap.add_argument("--n_featured", type=int, default=24, help="patches embedded in the HTML dashboard")
    args = ap.parse_args()

    thr = RiskThresholds()
    if args.thresholds and os.path.exists(args.thresholds):
        cfg = json.load(open(args.thresholds))
        thr = RiskThresholds(cfg["prob_moderate"], cfg["prob_high"], cfg["prob_extreme"], cfg["uncertainty_bump"])
        print("using calibrated thresholds:", thr)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=32)
    x_test = test_loader.dataset.inputs          # (N, 12, 64, 64) float32, normalized
    y_test = test_loader.dataset.labels[:, 0]    # (N, 64, 64) float32 in {0, 1}
    n_test = len(x_test)
    print(f"test patches: {n_test}")

    attn = load_model("attention", args.attn_ckpt, device)
    light = load_model("lightweight", args.light_ckpt, device)

    # ---- deterministic predictions on the full test split ----
    print("predicting (deterministic)...")
    p_attn = predict_all(attn, x_test, device)
    p_light = predict_all(light, x_test, device)

    metrics = {
        "meta": {
            "n_test_patches": int(n_test),
            "fire_pixel_fraction": round(float(y_test.mean()), 6),
            "mc_samples": args.mc_samples,
            "default_thresholds": vars(thr),
            "thresholds_calibrated": bool(args.thresholds),
            "tiers": RISK_TIERS,
        },
        "attention": {"accuracy": accuracy_metrics(p_attn.reshape(-1), y_test.reshape(-1))},
        "lightweight": {"accuracy": accuracy_metrics(p_light.reshape(-1), y_test.reshape(-1))},
    }

    ths = [round(0.05 * k, 2) for k in range(1, 20)]
    metrics["f1_sweep"] = {"thresholds": ths,
                           "attention": f1_sweep(p_attn, y_test, ths),
                           "lightweight": f1_sweep(p_light, y_test, ths)}
    metrics["pr_curves"] = {"attention": pr_curve(p_attn, y_test),
                            "lightweight": pr_curve(p_light, y_test)}

    # ---- MC-Dropout on the full test split (lightweight) ----
    print(f"MC-Dropout ({args.mc_samples} samples) on full test split...")
    mc_mean, mc_std = mc_predict_all(light, x_test, device, args.mc_samples)
    metrics["lightweight_mc"] = {"accuracy": accuracy_metrics(mc_mean.reshape(-1), y_test.reshape(-1))}

    # Risk-tier calibration: how often are pixels in each tier actually fire?
    tiers = assign_risk_tier(mc_mean, mc_std, thr)
    cal = []
    for i, name in enumerate(RISK_TIERS):
        m = tiers == i
        n = int(m.sum())
        cal.append({"tier": name, "pixels": n,
                    "pct_of_pixels": round(100 * n / tiers.size, 3),
                    "observed_fire_rate": round(float(y_test[m].mean()), 4) if n else None})
    metrics["risk_calibration"] = cal

    # Is the uncertainty signal informative? Compare std on correct vs wrong pixels.
    wrong = (mc_mean >= 0.5) != (y_test >= 0.5)
    metrics["uncertainty_stats"] = {
        "mean_std_correct_pixels": round(float(mc_std[~wrong].mean()), 5),
        "mean_std_wrong_pixels": round(float(mc_std[wrong].mean()), 5),
        "fraction_pixels_bumped": round(float((mc_std >= thr.uncertainty_bump).mean()), 5),
    }

    # ---- choose demo patches (mostly fire patches, a few quiet ones) ----
    rng = np.random.default_rng(0)
    fire_frac = y_test.reshape(n_test, -1).mean(1)
    fire_idx = np.where(fire_frac > 0)[0]
    quiet_idx = np.where(fire_frac == 0)[0]
    n_quiet = min(len(quiet_idx), max(args.n_patches // 10, 1))
    n_fire = min(len(fire_idx), args.n_patches - n_quiet)
    sel = np.concatenate([rng.choice(fire_idx, n_fire, replace=False),
                          rng.choice(quiet_idx, n_quiet, replace=False)])
    sel = np.sort(sel)
    # featured = evenly spaced by fire fraction among selected (variety for the dashboard)
    order = np.argsort(fire_frac[sel])
    pos = np.linspace(0, len(sel) - 1, min(args.n_featured, len(sel))).astype(int)
    featured = order[pos]

    np.savez_compressed(
        os.path.join(args.out_dir, "demo_bundle.npz"),
        x=x_test[sel].astype(np.float16),
        y=y_test[sel].astype(np.uint8),
        att=p_attn[sel].astype(np.float16),
        mc_mean=mc_mean[sel].astype(np.float16),
        mc_std=mc_std[sel].astype(np.float16),
        orig_idx=sel,
        featured=featured,
    )
    print(f"saved bundle with {len(sel)} patches ({len(featured)} featured)")

    # ---- efficiency: GPU (if any) and CPU ----
    print("efficiency benchmarks...")
    metrics["attention"]["efficiency"] = efficiency_metrics(attn, device)
    metrics["lightweight"]["efficiency"] = efficiency_metrics(light, device)
    cpu = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    a_cpu = load_model("attention", args.attn_ckpt, cpu)
    l_cpu = load_model("lightweight", args.light_ckpt, cpu)
    metrics["attention"]["efficiency_cpu"] = efficiency_metrics(a_cpu, cpu, n_latency_runs=50)
    metrics["lightweight"]["efficiency_cpu"] = efficiency_metrics(l_cpu, cpu, n_latency_runs=50)

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    shutil.copy(args.attn_ckpt, os.path.join(args.out_dir, "attention_best.pt"))
    shutil.copy(args.light_ckpt, os.path.join(args.out_dir, "lightweight_best.pt"))

    print("\n=== summary ===")
    for k in ("attention", "lightweight"):
        print(k, metrics[k]["accuracy"], metrics[k]["efficiency"])
    print("lightweight (MC mean):", metrics["lightweight_mc"]["accuracy"])
    print("risk calibration:", cal)
    print("uncertainty:", metrics["uncertainty_stats"])
    print(f"\nDone. Download the '{args.out_dir}' folder.")


if __name__ == "__main__":
    main()
