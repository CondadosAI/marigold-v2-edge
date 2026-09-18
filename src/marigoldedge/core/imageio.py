"""Image in, depth visualisation out.

Marigold's evaluation config is named ``..._768.yaml`` and its prompt
embedding ``realimg512``, so resolution is a real variable here rather than a
detail. Latent geometry constrains it: the Qwen VAE downsamples by 8 and the
DiT packs 2x2 latent patches, so the pixel dimensions must be multiples of 16.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

LATENT_MULTIPLE = 16


def load_rgb(path: Path, resolution: int | None = None) -> tuple[torch.Tensor, tuple[int, int]]:
    """Read an image as [1, 3, H, W] in [-1, 1], plus its original (W, H).

    ``resolution`` sets the long edge; the short edge follows the aspect ratio
    and both are rounded to a multiple of 16. Passing ``None`` keeps the native
    size, still rounded.
    """
    image = Image.open(path).convert("RGB")
    original = image.size
    width, height = image.size

    if resolution is not None:
        scale = resolution / max(width, height)
        width, height = round(width * scale), round(height * scale)

    width = max(LATENT_MULTIPLE, round(width / LATENT_MULTIPLE) * LATENT_MULTIPLE)
    height = max(LATENT_MULTIPLE, round(height / LATENT_MULTIPLE) * LATENT_MULTIPLE)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.LANCZOS)

    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0), original


def depth_to_array(depth: torch.Tensor) -> np.ndarray:
    """[1, 1, H, W] tensor in [-1, 1] -> float32 [H, W]. The saved artifact."""
    return depth[0, 0].float().cpu().numpy()


def colorize(depth: np.ndarray, cmap: str = "Spectral_r") -> Image.Image:
    """Percentile-normalised colour map, matching upstream's ``depth_spectral``.

    Normalising on the 2nd/98th percentile rather than min/max keeps one hot
    pixel from flattening the whole map, which matters when comparing backbones
    whose outliers differ.
    """
    import matplotlib

    low, high = np.percentile(depth, [2, 98])
    normalised = np.clip((depth - low) / max(high - low, 1e-8), 0, 1)
    colors = matplotlib.colormaps[cmap](normalised)[..., :3]
    return Image.fromarray((colors * 255).astype(np.uint8))
