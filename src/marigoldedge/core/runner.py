"""Assemble a runnable Marigold V2 from a backbone choice, and time it."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage
from loguru import logger

from marigoldedge import config
from marigoldedge.core import backends, pipeline, weights


@dataclass
class Timing:
    """Latency and memory for one configuration.

    ``warmup`` runs are discarded: the first call through a GGUF backbone pays
    for dequantization kernels and allocator growth that no later call repeats,
    and reporting it would flatter every other row by comparison.
    """

    seconds: list[float] = field(default_factory=list)
    peak_vram_gb: float = 0.0

    @property
    def median(self) -> float:
        return float(sorted(self.seconds)[len(self.seconds) // 2])


class MarigoldRunner:
    """A backbone, the Marigold LoRA, and the fine-tuned VAE decoder."""

    def __init__(self, transformer, vae, prompt, label: str, device: str = "cuda"):
        self.transformer = transformer
        self.vae = vae
        self.prompt = prompt
        self.label = label
        self.device = device

    @classmethod
    def build(cls, backend: str, gguf_path: Path | None = None, device: str = "cuda",
              blocks_per_group: int = backends.DEFAULT_BLOCKS_PER_GROUP,
              use_stream: bool = False, offload: bool = True):
        """Load every piece. The LoRA and the decoder are identical across backends.

        Order matters: the backbone is loaded on the host, the LoRA is written
        into it there, and only then is offloading applied. Offloading first
        would leave the adapter's weights outside the hooks that stream
        everything else, which fails quietly rather than loudly.
        """
        if backend == "gguf":
            if gguf_path is None:
                raise ValueError("backend 'gguf' needs --gguf-path")
            transformer = backends.load_gguf(gguf_path)
            label = gguf_path.stem.replace("Qwen-Image-Edit-2509-", "")
        elif backend == "nf4":
            transformer = backends.load_nf4(device=device)
            label = "bnb-NF4"
        elif backend == "bf16":
            transformer = backends.load_bf16()
            label = "bf16"
        else:
            raise ValueError(f"unknown backend: {backend}")

        lora_state, vae_decoder, rank, _ = weights.split_trainables(config.TRAINABLES)
        backends.load_lora_(transformer, lora_state)
        logger.info("backend={} lora_rank={}", label, rank)

        # NF4 is already placed on the device by bitsandbytes. The others either
        # stream from host RAM (a card too small to hold them) or go straight on
        # (a card that fits them). Which one is in force changes what the timing
        # measures -- PCIe bandwidth or the GPU -- so it is recorded per run.
        if backend != "nf4":
            if offload:
                backends.offload(transformer, device, blocks_per_group, use_stream)
            else:
                logger.info("no offload: placing the backbone directly on {}", device)
                transformer.to(device)

        vae = AutoencoderKLQwenImage.from_pretrained(
            str(config.QWEN_VAE_DIR), torch_dtype=torch.bfloat16
        )
        weights.load_vae_decoder_(vae, vae_decoder)
        vae = vae.to(device)

        prompt = pipeline.PromptConditioning.load(
            config.PROMPT_EMBEDS, config.PROMPT_MASK, device
        )
        return cls(transformer, vae, prompt, label, device)

    @torch.no_grad()
    def predict(self, rgb_norm: torch.Tensor) -> torch.Tensor:
        """Predict depth. The generator is re-seeded every call so that two
        backends see the identical VAE sample and differ only in the DiT."""
        generator = torch.Generator(device=self.device).manual_seed(pipeline.SEED)
        return pipeline.predict_depth(
            rgb_norm.to(self.device), self.transformer, self.vae, self.prompt, generator
        )

    def benchmark(self, rgb_norm: torch.Tensor, runs: int = 5, warmup: int = 2) -> Timing:
        """Median of ``runs`` timed passes after ``warmup`` discarded ones.

        Timing covers encode + step + decode, which is what a caller waits for.
        Model loading and image I/O are excluded and reported separately.
        """
        timing = Timing()
        torch.cuda.reset_peak_memory_stats()

        for _ in range(warmup):
            self.predict(rgb_norm)
        torch.cuda.synchronize()

        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            self.predict(rgb_norm)
            torch.cuda.synchronize()
            timing.seconds.append(time.perf_counter() - start)

        timing.peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        logger.info(
            "{}: {:.2f} s median over {} runs, peak VRAM {:.2f} GB",
            self.label, timing.median, runs, timing.peak_vram_gb,
        )
        return timing
