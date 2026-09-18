"""Tests for the checkpoint split.

This is the claim the whole project rests on: that the release is a LoRA over a
frozen backbone plus a VAE decoder. The split feeds both the article's anatomy
table and the loader, so a mistake here is expensive in two directions at once.

Everything runs on a synthetic checkpoint a few kilobytes wide. Nothing here
needs the real 1.85 GB file, a network, or a GPU.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from marigoldedge.core import weights

RANK = 128
DIM = 8


def _fake_checkpoint(path, *, blocks: int = 3, with_training_only: bool = True):
    """A checkpoint shaped like Marigold's, three orders of magnitude smaller."""
    tensors: dict[str, torch.Tensor] = {}
    for b in range(blocks):
        for proj in ("to_q", "to_k"):
            base = f"Diffuser.transformer_blocks.{b}.attn.{proj}"
            tensors[f"{base}.lora_A.default.weight"] = torch.zeros(RANK, DIM)
            tensors[f"{base}.lora_B.default.weight"] = torch.zeros(DIM, RANK)
    tensors["VAE.decoder.conv_in.weight"] = torch.zeros(DIM, DIM)
    tensors["VAE.post_quant_conv.bias"] = torch.zeros(DIM)
    if with_training_only:
        tensors["iREPAStudentProjector.out__hidden_-1.weight"] = torch.zeros(DIM, DIM)
    save_file(tensors, str(path))
    return tensors


def test_split_sorts_every_tensor_into_exactly_one_bucket(tmp_path):
    path = tmp_path / "trainables.safetensors"
    raw = _fake_checkpoint(path)

    lora, vae, rank, training_only = weights.split_trainables(path)

    assert len(lora) + len(vae) + len(training_only) == len(raw)
    assert len(lora) == 12  # 3 blocks x 2 projections x (A, B)
    assert len(vae) == 2
    assert len(training_only) == 1


def test_split_strips_the_prefix_but_keeps_the_peft_adapter_name(tmp_path):
    # The adapter name matters: these weights are loaded into a model that
    # already has an adapter called "default" attached, so stripping it would
    # make every key miss.
    path = tmp_path / "trainables.safetensors"
    _fake_checkpoint(path)

    lora, vae, _, _ = weights.split_trainables(path)

    assert all(not k.startswith("Diffuser.") for k in lora)
    assert any(".lora_A.default.weight" in k for k in lora)
    assert all(not k.startswith("VAE.") for k in vae)
    assert "decoder.conv_in.weight" in vae


def test_rank_is_read_off_the_tensor_not_assumed(tmp_path):
    path = tmp_path / "trainables.safetensors"
    _fake_checkpoint(path)

    _, _, rank, _ = weights.split_trainables(path)

    assert rank == RANK


def test_a_checkpoint_with_no_lora_is_an_error_not_a_zero(tmp_path):
    # A file that is not a Marigold checkpoint should fail loudly. Returning an
    # empty adapter would load cleanly and predict nonsense.
    path = tmp_path / "not_marigold.safetensors"
    save_file({"VAE.decoder.conv_in.weight": torch.zeros(DIM, DIM)}, str(path))

    with pytest.raises(ValueError, match="no lora_A tensor"):
        weights.split_trainables(path)


def test_a_missing_checkpoint_names_the_command_that_fetches_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="marigold-edge fetch"):
        weights.split_trainables(tmp_path / "absent.safetensors")


def test_loading_a_decoder_the_model_does_not_have_is_an_error():
    """A renamed layer upstream must fail here rather than predict quietly."""

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Linear(DIM, DIM, bias=False)

    with pytest.raises(ValueError, match="not present in the model"):
        weights.load_vae_decoder_(Stub(), {"decoder.nonexistent.weight": torch.zeros(DIM)})
