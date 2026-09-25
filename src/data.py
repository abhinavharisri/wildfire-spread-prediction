"""
Module 1: Data Preprocessing
=============================
Loads the Next Day Wildfire Spread (NDWS) dataset (Huot et al., 2022),
parses the TFRecord shards Kaggle distributes it as, clips/normalizes each
of the 12 input channels, and exposes PyTorch DataLoaders for train/val/test.

Dataset home: https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread
Download it with (requires a free Kaggle account + API token, see README):
    kaggle datasets download -d fantineh/next-day-wildfire-spread -p data/raw
    unzip data/raw/next-day-wildfire-spread.zip -d data/raw
"""
import glob
import os

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader

# The 12 input channels used by the base paper (Huot et al., 2022) and by
# every follow-up model we reviewed (Attention U-Net, LinU-Mamba, etc.).
INPUT_FEATURES = [
    "elevation", "th", "vs", "tmmn", "tmmx", "sph",
    "pr", "pdsi", "NDVI", "population", "erc", "PrevFireMask",
]
OUTPUT_FEATURE = "FireMask"
DATA_SIZE = 64  # 64x64 patches, as released

# Per-channel clip ranges from Huot et al. (2022), Table 1 — clips extreme
# outliers before normalization so a handful of bad pixels don't blow up
# the per-channel mean/std.
CLIP_RANGES = {
    "elevation": (0.0, 3141.0),
    "th": (0.0, 360.0),
    "vs": (0.0, 10.024),
    "tmmn": (253.15, 298.15),
    "tmmx": (253.15, 315.15),
    "sph": (0.0, 1.0),
    "pr": (0.0, 44.53),
    "pdsi": (-6.13, 7.88),
    "NDVI": (-1.0, 1.0),
    "population": (0.0, 2649.0),
    "erc": (0.0, 106.24),
    "PrevFireMask": (-1.0, 1.0),
}

_FEATURE_DESCRIPTION = {
    name: tf.io.FixedLenFeature([DATA_SIZE * DATA_SIZE], tf.float32)
    for name in INPUT_FEATURES + [OUTPUT_FEATURE]
}


def _parse_example(example_proto):
    parsed = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION)
    inputs = tf.stack(
        [tf.reshape(parsed[f], (DATA_SIZE, DATA_SIZE)) for f in INPUT_FEATURES],
        axis=0,
    )  # (12, 64, 64)
    label = tf.reshape(parsed[OUTPUT_FEATURE], (1, DATA_SIZE, DATA_SIZE))  # (1, 64, 64)
    return inputs, label


def _normalize(inputs: np.ndarray) -> np.ndarray:
    """Clip each channel to its known range, then z-score normalize."""
    out = inputs.copy()
    for i, name in enumerate(INPUT_FEATURES):
        lo, hi = CLIP_RANGES[name]
        out[i] = np.clip(out[i], lo, hi)
        mean, std = out[i].mean(), out[i].std() + 1e-6
        out[i] = (out[i] - mean) / std
    return out


class NDWSDataset(Dataset):
    """Materializes a TFRecord shard set into an in-memory PyTorch dataset.

    NDWS patches are small (64x64x13 floats ~= 213KB each) and the full
    train split is ~15-16K examples, so it comfortably fits in RAM (~3GB) —
    no need for a streaming TFRecord pipeline at train time.
    """

    def __init__(self, tfrecord_paths, normalize: bool = True):
        if not tfrecord_paths:
            raise FileNotFoundError(
                "No TFRecord shards found. Did you download and unzip the "
                "dataset? See README_IMPLEMENTATION.md."
            )
        raw_ds = tf.data.TFRecordDataset(tfrecord_paths)  # NDWS shards are uncompressed
        parsed_ds = raw_ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)

        self.inputs, self.labels = [], []
        for x, y in parsed_ds:
            x = x.numpy()
            if normalize:
                x = _normalize(x)
            self.inputs.append(x.astype(np.float32))
            # FireMask: 1 = fire, 0 = no fire, -1 = uncertain/masked -> treat as 0
            y = y.numpy().astype(np.float32)
            y = np.where(y < 0, 0.0, y)
            self.labels.append(y)

        self.inputs = np.stack(self.inputs)
        self.labels = np.stack(self.labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.labels[idx])


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    """Returns (train_loader, val_loader, test_loader).

    Expects data_dir to contain the unzipped Kaggle release, i.e. files
    matching `next_day_wildfire_spread_train_*.tfrecord`,
    `..._eval_*.tfrecord`, and `..._test_*.tfrecord`.
    """
    def shards(split):
        pattern = os.path.join(data_dir, f"*{split}*.tfrecord*")
        return sorted(glob.glob(pattern))

    train_ds = NDWSDataset(shards("train"))
    val_ds = NDWSDataset(shards("eval"))
    test_ds = NDWSDataset(shards("test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sanity-check the NDWS data pipeline.")
    ap.add_argument("--data_dir", default="data/raw")
    args = ap.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, batch_size=8)
    xb, yb = next(iter(train_loader))
    print(f"train batches: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
    print(f"input batch shape:  {tuple(xb.shape)}  (expect [B, 12, 64, 64])")
    print(f"label batch shape:  {tuple(yb.shape)}  (expect [B, 1, 64, 64])")
    print(f"fire-pixel fraction in this batch: {yb.mean().item():.4%}")
