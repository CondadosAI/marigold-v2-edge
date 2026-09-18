#!/usr/bin/env python3
"""Fetch every weight the benchmark needs, on the pod rather than over a home line.

Three downloads, ~56 GB total:

* the bf16 Qwen-Image-Edit-2509 transformer (40.9 GB) — serves two rows at
  once, because bitsandbytes quantizes NF4 from it on load rather than from a
  separate checkpoint;
* one GGUF level (Q4_K_M, 13.1 GB) — the same file the laptop ran, so the two
  machines are comparable;
* Marigold's own 1.85 GB checkpoint, its prompt embeddings, and the stock VAE.

Run it before the benchmark, not inside it: a download that fails halfway
should not look like a benchmark that failed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = Path(os.environ.get("MARIGOLD_EDGE_MODELS_DIR", "/workspace/models"))
MARIGOLD_REPO = "huawei-bayerlab/marigold-v2-0"
QWEN_REPO = "Qwen/Qwen-Image-Edit-2509"
GGUF_REPO = "QuantStack/Qwen-Image-Edit-2509-GGUF"
GGUF_FILE = "Qwen-Image-Edit-2509-Q4_K_M.gguf"
EMBED_PREFIX = "qwen_edit_2509_qwen_depth_realimg512"


def assert_gpu() -> None:
    """A device that is listed is not a device that loaded."""
    import torch

    assert torch.cuda.is_available(), "no CUDA"
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    assert arch in torch.cuda.get_arch_list(), (
        f"{arch} not in this build's arch list {torch.cuda.get_arch_list()}"
    )
    free, total = torch.cuda.mem_get_info(0)
    print(f"=== GPU: {torch.cuda.get_device_name(0)} {arch} "
          f"{total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
    print(f"=== torch {torch.__version__}")
    if total < 45e9:
        sys.exit(f"need ~45 GB for the bf16 row; this card has {total / 1e9:.1f} GB")


def main() -> None:
    assert_gpu()
    MODELS.mkdir(parents=True, exist_ok=True)

    print("=== Marigold checkpoint + prompt embeddings")
    snapshot_download(
        MARIGOLD_REPO,
        local_dir=MODELS / "marigold",
        allow_patterns=[
            "depth/Log-stage2/trainables.safetensors",
            f"qwen_text_embeddings/{EMBED_PREFIX}_prompt_*.pt",
            "manifest.json",
        ],
        max_workers=4,
    )

    print("=== stock Qwen VAE")
    snapshot_download(
        QWEN_REPO, local_dir=MODELS / "qwen-vae-src",
        allow_patterns=["vae/*"], max_workers=4,
    )
    vae_dir = MODELS / "qwen-vae"
    vae_dir.mkdir(exist_ok=True)
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        target = vae_dir / name
        if not target.exists():
            target.symlink_to(MODELS / "qwen-vae-src" / "vae" / name)

    print(f"=== GGUF {GGUF_FILE}")
    gguf_dir = MODELS / "gguf"
    gguf_dir.mkdir(exist_ok=True)
    path = hf_hub_download(GGUF_REPO, GGUF_FILE, local_dir=gguf_dir)
    print("    ", path, Path(path).stat().st_size, "bytes")

    print("=== bf16 transformer (40.9 GB; this is the long one)")
    snapshot_download(QWEN_REPO, allow_patterns=["transformer/*"], max_workers=8)

    print("=== DONE fetch")


if __name__ == "__main__":
    main()
