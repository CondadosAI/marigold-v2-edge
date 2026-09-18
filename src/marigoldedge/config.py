"""Project-root-aware paths. Every one is overridable by environment variable."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
DATA_DIR = Path(os.environ.get("MARIGOLD_EDGE_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("MARIGOLD_EDGE_MODELS_DIR", PROJECT_ROOT / "models"))
OUTPUT_DIR = Path(os.environ.get("MARIGOLD_EDGE_OUTPUT_DIR", PROJECT_ROOT / "output"))

MARIGOLD_DIR = MODELS_DIR / "marigold"
TRAINABLES = MARIGOLD_DIR / "depth" / "Log-stage2" / "trainables.safetensors"
EMBED_DIR = MARIGOLD_DIR / "qwen_text_embeddings"
EMBED_PREFIX = "qwen_edit_2509_qwen_depth_realimg512"
PROMPT_EMBEDS = EMBED_DIR / f"{EMBED_PREFIX}_prompt_embeds.pt"
PROMPT_MASK = EMBED_DIR / f"{EMBED_PREFIX}_prompt_mask.pt"
QWEN_VAE_DIR = MODELS_DIR / "qwen-vae"
GGUF_DIR = MODELS_DIR / "gguf"
