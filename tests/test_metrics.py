"""Tests for the fidelity metrics.

Every comparison the article publishes goes through `align_affine` first. If it
is wrong, every fidelity number is wrong in the same direction, which is the
kind of error that looks like a finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from marigoldedge.core import metrics

RNG = np.random.default_rng(2026)


def _depth(shape=(32, 40)):
    """Something shaped like Marigold's output: smooth, and straddling zero."""
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.sin(x / 7.0) + np.cos(y / 5.0)


def test_alignment_recovers_a_known_scale_and_shift():
    target = _depth()
    scaled = 3.5 * target - 1.25

    aligned = metrics.align_affine(scaled, target)

    np.testing.assert_allclose(aligned, target, atol=1e-10)


def test_alignment_is_a_no_op_when_the_arrays_already_agree():
    target = _depth()

    np.testing.assert_allclose(metrics.align_affine(target, target), target, atol=1e-10)


def test_alignment_needs_more_than_one_valid_pixel():
    with pytest.raises(ValueError, match="not enough valid pixels"):
        metrics.align_affine(np.array([1.0]), np.array([1.0]))


def test_identical_maps_score_perfectly():
    target = _depth()

    scores = metrics.compare_to_reference(target, target)

    assert scores["rmse_vs_ref"] == pytest.approx(0.0, abs=1e-10)
    assert scores["abs_rel_vs_ref"] == pytest.approx(0.0, abs=1e-10)
    assert scores["delta1_vs_ref"] == pytest.approx(1.0)


def test_a_map_that_differs_only_by_scale_scores_perfectly_too():
    # This is the property that makes the metric meaningful for affine-invariant
    # depth: a prediction is not penalised for a freedom it was never asked to
    # resolve.
    target = _depth()

    scores = metrics.compare_to_reference(2.0 * target + 5.0, target)

    assert scores["rmse_vs_ref"] == pytest.approx(0.0, abs=1e-10)


def test_noise_degrades_rmse_monotonically():
    target = _depth()
    a = metrics.compare_to_reference(target + RNG.normal(0, 0.01, target.shape), target)
    b = metrics.compare_to_reference(target + RNG.normal(0, 0.10, target.shape), target)

    assert a["rmse_vs_ref"] < b["rmse_vs_ref"]


def test_abs_rel_blows_up_near_zero_which_is_why_the_article_does_not_lead_with_it():
    """Pin the pathology the post describes, so it stays described.

    Two maps that differ by a constant small error score very differently under
    AbsRel depending only on how close the target sits to zero. The same pair
    under RMSE differs not at all.
    """
    error = 0.01
    near_zero = np.full((16, 16), 0.02)
    far_from_zero = np.full((16, 16), 2.0)

    near = metrics.abs_rel(near_zero + error, near_zero)
    far = metrics.abs_rel(far_from_zero + error, far_from_zero)

    assert near > 50 * far
