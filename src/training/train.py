"""
Module 4: Training & Tuning
=============================
Trains either the baseline AttentionUNet or the LightweightUNet on NDWS.

Usage:
    python -m src.training.train --model attention --epochs 30 --data_dir data/raw
    python -m src.training.train --model lightweight --epochs 30 --data_dir data/raw

Loss: since fire pixels are a small minority of every 64x64 patch (severe
class imbalance, the same issue flagged in the credit-card-fraud reference
deck for 0.17% fraud rate), plain BCE would let the model collapse to
"predict no fire everywhere" and still score >95% pixel accuracy. We use
Focal Loss (Lin et al., 2017) instead, which down-weights easy (already
well-classified) pixels and concentrates gradient on the hard, usually-fire
pixels.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data import get_dataloaders
from src.models.attention_unet import AttentionUNet
from src.models.lightweight_unet import LightweightUNet


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # weight on the positive (fire) class
        self.gamma = gamma

    def forward(self, pred, target):
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        pos_loss = -self.alpha * (1 - pred) ** self.gamma * target * torch.log(pred)
        neg_loss = -(1 - self.alpha) * pred ** self.gamma * (1 - target) * torch.log(1 - pred)
        return (pos_loss + neg_loss).mean()


def build_model(name: str) -> nn.Module:
    if name == "attention":
        return AttentionUNet(in_channels=12, base_ch=32)
    if name == "lightweight":
        return LightweightUNet(in_channels=12, base_ch=24, dropout_p=0.2)
    raise ValueError(f"unknown model {name!r}, expected 'attention' or 'lightweight'")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["attention", "lightweight"], required=True)
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--patience", type=int, default=6, help="early stopping patience (epochs)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, val_loader, _ = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}  |  parameters: {n_params:,}")

    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss, epochs_no_improve = float("inf"), 0
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)
        dt = time.time() - t0
        print(f"epoch {epoch:03d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({dt:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"  -> saved new best checkpoint to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping: no val improvement in {args.patience} epochs")
                break

    print(f"done. best val_loss={best_val_loss:.4f}, checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
