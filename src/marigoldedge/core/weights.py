"""Split Marigold V2's ``trainables.safetensors`` into the parts each runtime needs.

The released checkpoint is not a model. It is three things concatenated:

* ``Diffuser.*``  — a rank-128 PEFT LoRA over the frozen Qwen-Image-Edit-2509
  MMDiT (attention + MLP projections in all 60 blocks, plus ``img_in`` and
  ``norm_out.linear``).
* ``VAE.*``       — full weights for a fine-tuned VAE *decoder*. This is what
  turns latents into a depth map rather than an image, so it is not optional
  and it is not a LoRA.
* ``iREPA*``      — the REPA student projector, used by the training loss only.
  Dropping it at inference is correct, not a shortcut.

Keeping the split explicit is what lets the same LoRA ride on a bf16, a
bitsandbytes-NF4 or a GGUF backbone without touching the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

DIT_PREFIX = "Diffuser."
VAE_PREFIX = "VAE."


def split_trainables(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, dict[str, torch.Tensor]]:
    """Return ``(lora_state_dict, vae_decoder_state_dict, lora_rank, training_only)``.

    ``lora_state_dict`` keeps PEFT's own key convention (including the
    ``.default`` adapter name) because it is loaded straight into a model that
    already has an adapter of that name attached. Renaming it to diffusers'
    ``load_lora_adapter`` convention would be a second, lossier path to the
    same place.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Marigold checkpoint not found at {path}.\n"
            "It is 1.85 GB and is not committed to this repository. Fetch it with:\n"
            "    uv run marigold-edge fetch"
        )
    raw = load_file(str(path))
    lora: dict[str, torch.Tensor] = {}
    vae: dict[str, torch.Tensor] = {}
    training_only: dict[str, torch.Tensor] = {}

    for key, tensor in raw.items():
        if key.startswith(DIT_PREFIX):
            lora[key[len(DIT_PREFIX) :]] = tensor
        elif key.startswith(VAE_PREFIX):
            vae[key[len(VAE_PREFIX) :]] = tensor
        else:
            training_only[key] = tensor  # iREPAStudentProjector

    rank = _infer_rank(lora)
    logger.info(
        "trainables: {} LoRA tensors (rank {}), {} VAE-decoder tensors, {} training-only dropped",
        len(lora),
        rank,
        len(vae),
        len(training_only),
    )
    return lora, vae, rank, training_only


def _infer_rank(lora: dict[str, torch.Tensor]) -> int:
    """Read the rank off a ``lora_A`` tensor instead of trusting the config."""
    for key, tensor in lora.items():
        if ".lora_A." in key:
            return int(tensor.shape[0])
    raise ValueError("no lora_A tensor found; is this a Marigold V2 checkpoint?")


def load_vae_decoder_(vae, decoder_state: dict[str, torch.Tensor]) -> None:
    """Overwrite the stock Qwen decoder with Marigold's, in place.

    Loaded non-strictly because ``decoder_state`` covers only the decoder while
    ``vae`` also holds an encoder. Every *provided* key must land, though, so a
    renamed layer upstream fails loudly here instead of silently producing a
    plausible-looking wrong depth map.
    """
    missing, unexpected = vae.load_state_dict(decoder_state, strict=False)
    if unexpected:
        raise ValueError(f"Marigold VAE keys not present in the model: {unexpected[:5]}")
    logger.info(
        "VAE decoder swapped: {} tensors applied, {} model keys left untouched (encoder)",
        len(decoder_state),
        len(missing),
    )
