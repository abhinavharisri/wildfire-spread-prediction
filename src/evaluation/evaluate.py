"""
Module 5: Efficiency Benchmarking  +  Module 8: Evaluation & Visualization
=============================================================================
Produces the exact numbers the Review-1/2 "Proposed Method and Feasibility
Study" slide needs: accuracy (AUC-PR, F1, IoU) *and* efficiency (parameter
count, FLOPs, inference latency) for the baseline and lightweight model,
side by side. This head-to-head reporting is itself the project's Gap-1
contribution — none of the 11 reviewed papers report both together.

Usage:
    python -m src.evaluation.evaluate --model attention --ckpt checkpoints/attention_best.pt --data_dir data/raw
    python -m src.evaluation.evaluate --model lightweight --ckpt checkpoints/lightweight_best.pt --data_dir data/raw
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data import get_dataloaders
from src.training.train import build_model


def iou_score(pred_bin: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(pred_bin, target).sum()
    union = np.logical_or(pred_bin, target).sum()
    return float(intersection / union) if union > 0 else 1.0


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        all_pred.append(pred.reshape(-1))
        all_true.append(y.numpy().reshape(-1))
    return np.concatenate(all_pred), np.concatenate(all_true)


def accuracy_metrics(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
    y_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = y_true.astype(np.uint8)
    return {
        "AUC-PR": round(float(average_precision_score(y_true_bin, y_pred)), 4),
        "F1": round(float(f1_score(y_true_bin, y_bin, zero_division=0)), 4),
        "IoU": round(iou_score(y_bin, y_true_bin), 4),
    }


def efficiency_metrics(model, device, input_shape=(1, 12, 64, 64), n_latency_runs=100) -> dict:
    n_params = sum(p.numel() for p in model.parameters())

    flops = None
    try:
        from thop import profile
        dummy = torch.randn(*input_shape).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    except ImportError:
        pass  # thop not installed -- FLOPs will be reported as None

    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):  # warm-up
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_latency_runs):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    latency_ms = 1000 * elapsed / n_latency_runs

    return {
        "parameters": n_params,
        "parameters_M": round(n_params / 1e6, 3),
        "FLOPs_G": round(flops / 1e9, 4) if flops else None,
        "latency_ms_per_image": round(latency_ms, 4),
        "device": str(device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["attention", "lightweight"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model_state"])

    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    y_pred, y_true = collect_predictions(model, test_loader, device)

    acc = accuracy_metrics(y_pred, y_true)
    eff = efficiency_metrics(model, device)

    print(f"\n=== {args.model} ===")
    print("Accuracy:  ", acc)
    print("Efficiency:", eff)
    print("\nCopy straight into the Review feasibility slide:")
    print(f"  ROC/AUC-PR: {acc['AUC-PR']}   F1: {acc['F1']}   IoU: {acc['IoU']}")
    print(f"  Parameters: {eff['parameters_M']}M   FLOPs: {eff['FLOPs_G']}G   "
          f"Latency: {eff['latency_ms_per_image']} ms/image on {eff['device']}")


if __name__ == "__main__":
    main()
