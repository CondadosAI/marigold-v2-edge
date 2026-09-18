"""Tests for image loading.

The size rule is a hard constraint from the model rather than a preference: the
Qwen VAE downsamples by 8 and the transformer packs 2x2 latent patches, so any
side that is not a multiple of 16 produces a latent grid the packer cannot
divide. Getting it wrong fails deep inside the transformer with a shape error
that names none of this.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from marigoldedge.core import imageio

MULTIPLE = 16


@pytest.fixture
def photo(tmp_path):
    def _make(width: int, height: int):
        path = tmp_path / f"{width}x{height}.png"
        Image.fromarray(
            np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
        ).save(path)
        return path

    return _make


@pytest.mark.parametrize(
    ("width", "height", "resolution"),
    [(1200, 900, 768), (1000, 1000, 512), (640, 480, None), (101, 97, 256)],
)
def test_both_sides_come_back_a_multiple_of_sixteen(photo, width, height, resolution):
    tensor, _ = imageio.load_rgb(photo(width, height), resolution)

    _, _, h, w = tensor.shape
    assert h % MULTIPLE == 0
    assert w % MULTIPLE == 0


def test_the_long_edge_follows_the_requested_resolution(photo):
    tensor, _ = imageio.load_rgb(photo(1200, 900), 768)

    _, _, h, w = tensor.shape
    assert max(h, w) == 768
    # 900 * (768/1200) = 576, already a multiple of 16
    assert (h, w) == (576, 768)


def test_aspect_ratio_survives_the_rounding(photo):
    tensor, _ = imageio.load_rgb(photo(1200, 900), 768)

    _, _, h, w = tensor.shape
    assert w / h == pytest.approx(1200 / 900, rel=0.02)


def test_pixels_arrive_in_the_range_the_vae_expects(photo):
    tensor, _ = imageio.load_rgb(photo(320, 240), 256)

    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0
    assert tensor.shape[1] == 3  # channels-first, RGB


def test_the_original_size_is_reported_unchanged(photo):
    _, original = imageio.load_rgb(photo(1201, 903), 768)

    assert original == (1201, 903)


def test_a_tiny_image_is_not_rounded_away_to_nothing(photo):
    # round(8 / 16) * 16 is 0, which would be a zero-sized tensor.
    tensor, _ = imageio.load_rgb(photo(8, 8), None)

    _, _, h, w = tensor.shape
    assert h >= MULTIPLE
    assert w >= MULTIPLE


def test_depth_to_array_drops_the_batch_and_channel_axes():
    import torch

    depth = torch.zeros(1, 1, 24, 32)

    array = imageio.depth_to_array(depth)

    assert array.shape == (24, 32)
    assert array.dtype == np.float32
