"""Marigold V2 depth, reimplemented against plain diffusers.

The upstream repository runs this through a config-driven registry, which is
excellent for training and awkward to point at a different backbone. The actual
inference graph is small enough to state directly:

    latents = normalise(VAE.encode(rgb).sample(generator))
    v       = DiT(pack(latents), t=0.499, frozen_prompt_embeds)
    depth   = VAE.decode(denormalise(latents - unpack(v))).mean(channels)

One property of that graph is what makes a quantized backbone worth trying:
there is a single function evaluation, not a schedule, so latency is one
forward pass and the 20.4 B parameters are read exactly once per image.

The graph is *not* fully deterministic. ``VAE.encode`` returns a distribution
and Marigold samples it, so the seed is part of the configuration; upstream
fixes it at 2026 and so does this code. Holding it constant is what makes two
backbones comparable at all.

Values here are transcribed from ``marigoldv2_inference.py`` in the authors'
Hugging Face Space and cross-checked against
``marigoldv2/experiments/20260316_qwen_depth/network_graph.py`` upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import QwenImageEditPipeline

# Fixed rectified-flow timestep, computed in bf16 exactly as during training.
TIMESTEP_RAW = 499.0
# The depth checkpoints predict velocity: x0 = x_t - v.
PREDICT_VELOCITY = True
# Upstream's seed. The VAE posterior is sampled, so this is load-bearing.
SEED = 2026
VAE_SCALE_FACTOR = 8


@dataclass(frozen=True)
class PromptConditioning:
    """The text encoder's frozen output, shipped as a tensor.

    Marigold precomputes this once and never loads Qwen2.5-VL at inference. The
    8.3 B text encoder is not part of the runtime at all, which is why the
    memory budget is the DiT plus the VAE and nothing else.
    """

    embeds: torch.Tensor
    mask: torch.Tensor

    @classmethod
    def load(cls, embeds_path, mask_path, device, dtype=torch.bfloat16) -> PromptConditioning:
        embeds = torch.load(embeds_path, map_location="cpu", weights_only=False)
        mask = torch.load(mask_path, map_location="cpu", weights_only=False)
        return cls(embeds=embeds[:1].to(device, dtype), mask=mask[:1].to(device) > 0)


def _latent_stats(vae, like: torch.Tensor):
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=like.device, dtype=like.dtype)
    std_inv = 1.0 / torch.tensor(vae.config.latents_std, device=like.device, dtype=like.dtype)
    return mean.view(shape), std_inv.view(shape)


@torch.no_grad()
def encode(rgb: torch.Tensor, vae, generator: torch.Generator) -> torch.Tensor:
    """RGB in [-1, 1], [B, 3, H, W] -> normalised latents [B, C, 1, h, w].

    The Qwen VAE is a video autoencoder, hence the ``unsqueeze(2)`` giving the
    single frame a temporal axis.
    """
    latents = vae.encode(rgb.to(vae.dtype).unsqueeze(2)).latent_dist.sample(generator)
    mean, std_inv = _latent_stats(vae, latents)
    return (latents - mean) * std_inv


@torch.no_grad()
def flow_step(latents: torch.Tensor, transformer, prompt: PromptConditioning) -> torch.Tensor:
    """The single rectified-flow step."""
    batch, channels, _, height, width = latents.shape
    packed = QwenImageEditPipeline._pack_latents(
        latents[:, :, 0], batch, channels, height, width
    ).to(torch.bfloat16)
    # bf16 arithmetic on purpose: this is how the timestep was computed during training.
    timestep = (
        torch.full((batch,), TIMESTEP_RAW, device=latents.device, dtype=torch.bfloat16) / 1000.0
    )
    velocity = transformer(
        hidden_states=packed,
        timestep=timestep,
        encoder_hidden_states=prompt.embeds,
        encoder_hidden_states_mask=prompt.mask,
        img_shapes=[[(1, height // 2, width // 2)]] * batch,
        return_dict=False,
    )[0]
    velocity = QwenImageEditPipeline._unpack_latents(
        velocity, height * VAE_SCALE_FACTOR, width * VAE_SCALE_FACTOR,
        vae_scale_factor=VAE_SCALE_FACTOR,
    )
    return latents - velocity.to(latents.dtype) if PREDICT_VELOCITY else velocity.to(latents.dtype)


@torch.no_grad()
def decode(latents: torch.Tensor, vae) -> torch.Tensor:
    """Normalised latents -> the decoded three-channel prediction, [B, 3, H, W]."""
    mean, std_inv = _latent_stats(vae, latents)
    return vae.decode(latents / std_inv + mean).sample[:, :, 0]


def to_depth(pixels: torch.Tensor) -> torch.Tensor:
    """Marigold's decoder emits three channels; the depth map is their mean.

    That is upstream's ``SelectChannel(channel="mean")``, not an approximation.
    """
    return pixels.mean(dim=1, keepdim=True)


@torch.no_grad()
def predict_depth(rgb: torch.Tensor, transformer, vae, prompt, generator) -> torch.Tensor:
    """Full graph: RGB in [-1, 1] -> relative depth, [B, 1, H, W]."""
    return to_depth(decode(flow_step(encode(rgb, vae, generator), transformer, prompt), vae))
