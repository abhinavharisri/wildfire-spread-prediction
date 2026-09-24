"""
Module 6: Uncertainty Estimation  +  Module 7: Risk Triage
=============================================================
This is the project's decision-support AI component (targets Gaps 3 and 5:
no reviewed paper pairs an efficient model with calibrated uncertainty, and
none convert a raw probability map into an actionable output).

Pipeline:
  1. Run T stochastic forward passes through the LightweightUNet with
     MC-Dropout enabled -> T slightly different probability maps per pixel.
  2. mean(T maps)  = the point prediction (spread probability)
     std(T maps)   = the uncertainty / disagreement signal
  3. Map (mean, std) -> one of four risk tiers per pixel via fixed
     thresholds, mirroring how the reference fraud-detection deck used a
     cost-based Approve/Review/Block triage instead of a single cutoff.

The thresholds below are a reasonable starting point, not the final word —
Section "Calibrating the thresholds" at the bottom explains how to tune
them once you have real validation predictions.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RiskThresholds:
    # Probability thresholds (mean prediction)
    prob_moderate: float = 0.3
    prob_high: float = 0.6
    prob_extreme: float = 0.85
    # Uncertainty threshold: predictions with std above this get bumped up
    # one tier regardless of mean probability, because a confident "High"
    # is safer to act on than an uncertain one.
    uncertainty_bump: float = 0.15


RISK_TIERS = ["Low", "Moderate", "High", "Extreme"]


@torch.no_grad()
def mc_dropout_predict(model, x: torch.Tensor, n_samples: int = 20):
    """Runs `n_samples` stochastic forward passes.

    Args:
        model: a LightweightUNet instance (must have been built with
            dropout_p > 0 and have `.enable_mc_dropout()`).
        x: input batch, (B, 12, 64, 64).
        n_samples: number of stochastic forward passes (T). 20-30 is a
            common choice in the MC-Dropout literature (Gal & Ghahramani,
            2016) — enough to stabilize the std estimate without making
            inference prohibitively slow.

    Returns:
        mean: (B, 1, 64, 64) — the averaged prediction (use this as the
            point estimate of next-day spread probability).
        std:  (B, 1, 64, 64) — per-pixel standard deviation across the T
            passes (the uncertainty signal).
    """
    model.enable_mc_dropout()
    samples = torch.stack([model(x) for _ in range(n_samples)], dim=0)  # (T, B, 1, 64, 64)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    return mean, std


def assign_risk_tier(mean: np.ndarray, std: np.ndarray, thresholds: RiskThresholds = RiskThresholds()) -> np.ndarray:
    """Per-pixel risk tier assignment.

    Args:
        mean, std: numpy arrays of the same shape (any shape — typically
            (B, 1, 64, 64) or (64, 64) for a single map), values in [0, 1].
    Returns:
        int array of the same shape, values 0-3 indexing into RISK_TIERS.
    """
    tier = np.zeros_like(mean, dtype=np.int64)
    tier[mean >= thresholds.prob_moderate] = 1
    tier[mean >= thresholds.prob_high] = 2
    tier[mean >= thresholds.prob_extreme] = 3

    # Uncertainty bump: a highly-disagreed-upon pixel is escalated one
    # tier (capped at Extreme) so it gets human attention rather than
    # being silently trusted.
    bump_mask = std >= thresholds.uncertainty_bump
    tier[bump_mask] = np.minimum(tier[bump_mask] + 1, len(RISK_TIERS) - 1)
    return tier


def risk_tier_report(mean: np.ndarray, std: np.ndarray, thresholds: RiskThresholds = RiskThresholds()) -> dict:
    """Summarizes one prediction map into the kind of readable line an
    emergency planner would want, mirroring the fraud-detection deck's
    'investigation summary' pattern."""
    tier = assign_risk_tier(mean, std)
    total = tier.size
    counts = {name: int((tier == i).sum()) for i, name in enumerate(RISK_TIERS)}
    pct = {name: round(100 * c / total, 2) for name, c in counts.items()}
    return {
        "tier_pixel_counts": counts,
        "tier_pixel_pct": pct,
        "max_tier": RISK_TIERS[int(tier.max())],
        "mean_confidence": round(float(1.0 - std.mean()), 4),
        "recommendation": _recommendation(RISK_TIERS[int(tier.max())]),
    }


def _recommendation(max_tier: str) -> str:
    return {
        "Low": "No action needed; continue routine monitoring.",
        "Moderate": "Monitor closely; alert local fire watch.",
        "High": "Prepare containment resources; consider pre-positioning crews.",
        "Extreme": "Escalate immediately to incident command; evacuation planning recommended.",
    }[max_tier]


# ---------------------------------------------------------------------------
# Calibrating the thresholds
# ---------------------------------------------------------------------------
# Once you have real validation-set predictions:
#   1. Run mc_dropout_predict over the validation set.
#   2. Sweep prob_moderate/prob_high/prob_extreme so that the *pixel-level*
#      precision within each tier roughly matches the tier's intent (e.g.
#      "Extreme" pixels should be fire >90% of the time in hindsight).
#   3. Sweep uncertainty_bump so that roughly 10-15% of pixels get bumped
#      -- too high and everything becomes "Extreme" and the tiering is
#      useless; too low and the uncertainty signal does nothing.
# This is exactly the kind of grid-search the fraud-detection reference
# deck did for its cost-optimized Approve/Review/Block thresholds.

if __name__ == "__main__":
    # Smoke test with random data (no trained weights needed) to confirm
    # the plumbing works end to end.
    import sys
    sys.path.insert(0, ".")
    from src.models.lightweight_unet import LightweightUNet

    model = LightweightUNet(in_channels=12, base_ch=24, dropout_p=0.2)
    x = torch.randn(1, 12, 64, 64)
    mean, std = mc_dropout_predict(model, x, n_samples=10)
    report = risk_tier_report(mean.numpy(), std.numpy())
    print("mean shape:", tuple(mean.shape), " std shape:", tuple(std.shape))
    print(report)
