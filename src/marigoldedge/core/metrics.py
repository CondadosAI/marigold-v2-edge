"""Depth metrics, computed the way monocular-depth papers compute them.

Marigold predicts *affine-invariant* relative depth: the prediction is only
defined up to a scale and a shift, so comparing two depth maps elementwise
without first resolving that freedom measures the freedom, not the error. Every
function here aligns before it scores.
"""

from __future__ import annotations

import numpy as np


def align_affine(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None):
    """Least-squares fit of ``a * pred + b`` to ``target``.

    This is the standard scale-and-shift alignment. Solving it in closed form
    on the valid pixels costs nothing and removes the one degree of freedom the
    model was never asked to pin down.
    """
    if mask is None:
        mask = np.isfinite(pred) & np.isfinite(target)
    p, t = pred[mask].astype(np.float64), target[mask].astype(np.float64)
    if p.size < 2:
        raise ValueError("not enough valid pixels to align")
    design = np.stack([p, np.ones_like(p)], axis=1)
    scale, shift = np.linalg.lstsq(design, t, rcond=None)[0]
    return scale * pred + shift


def abs_rel(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean absolute relative error. The headline number in every depth paper."""
    if mask is None:
        mask = np.isfinite(pred) & np.isfinite(target) & (np.abs(target) > 1e-6)
    p, t = pred[mask], target[mask]
    return float(np.mean(np.abs(p - t) / np.abs(t)))


def delta1(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of pixels within a 1.25x ratio of the target. Higher is better.

    **This is not the delta-1 that depth papers report, and the two must not be
    compared.** The standard metric assumes metric depth, which is strictly
    positive, so the ratio of prediction to target is well defined. Marigold's
    output is affine-invariant and straddles zero, so this implementation shifts
    both arrays by a common offset to make the ratio computable at all. The
    offset depends on the data, which means the number is not comparable across
    images, let alone against a published benchmark.

    It is kept because it moves in the right direction and is cheap, and it is
    written down here because the value ends up in the saved artifacts where
    somebody could mistake it for the real thing. For judging how close two
    depth maps are, prefer the RMSE and correlation in `compare_to_reference`;
    the same caveat applies to `abs_rel` above, which is why the article reports
    neither as its headline.
    """
    if mask is None:
        mask = np.isfinite(pred) & np.isfinite(target) & (np.abs(target) > 1e-6)
    p, t = pred[mask], target[mask]
    # Shift both into a strictly positive range so the ratio is meaningful for
    # relative depth, which is free to straddle zero.
    offset = min(p.min(), t.min())
    if offset <= 0:
        p, t = p - offset + 1e-3, t - offset + 1e-3
    ratio = np.maximum(p / t, t / p)
    return float(np.mean(ratio < 1.25))


def compare_to_reference(pred: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Score a quantized backbone against the bf16 run on the same image.

    This isolates the cost of quantization. It says nothing about whether
    Marigold is right, only about how far the cheap backbone drifted from the
    expensive one -- which is the question a reader choosing a quantization
    level actually has.
    """
    aligned = align_affine(pred, reference)
    return {
        "abs_rel_vs_ref": abs_rel(aligned, reference),
        "delta1_vs_ref": delta1(aligned, reference),
        "rmse_vs_ref": float(np.sqrt(np.mean((aligned - reference) ** 2))),
    }
