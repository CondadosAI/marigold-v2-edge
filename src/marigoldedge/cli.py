"""Command line for the Marigold-V2-on-6-GB benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from loguru import logger

from marigoldedge import config
from marigoldedge.core import imageio, metrics

MARIGOLD_REPO = "huawei-bayerlab/marigold-v2-0"


@click.group()
def cli() -> None:
    """Run and benchmark Marigold V2 depth on a quantized backbone."""


@cli.command()
@click.option("--image", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--backend", type=click.Choice(["gguf", "nf4", "bf16"]), default="gguf")
@click.option("--gguf-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--resolution", type=int, default=768, help="Long edge in pixels.")
@click.option("--offload/--no-offload", default=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=config.OUTPUT_DIR)
def predict(image: Path, backend: str, gguf_path: Path | None, resolution: int,
            offload: bool, out_dir: Path):
    """Predict one depth map and save the raw array next to a colour preview."""
    from marigoldedge.core.runner import MarigoldRunner

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, _ = imageio.load_rgb(image, resolution)
    runner = MarigoldRunner.build(backend, gguf_path, offload=offload)
    depth = imageio.depth_to_array(runner.predict(rgb))

    stem = f"{image.stem}_{runner.label}_{resolution}"
    np.save(out_dir / f"{stem}.npy", depth)
    imageio.colorize(depth).save(out_dir / f"{stem}.png")
    logger.info("wrote {}.npy and {}.png", stem, stem)


@cli.command()
@click.option("--image", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--backend", type=click.Choice(["gguf", "nf4", "bf16"]), default="gguf")
@click.option("--gguf-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--resolution", type=int, default=768)
@click.option("--runs", type=int, default=5)
@click.option("--warmup", type=int, default=2)
@click.option("--reference", type=click.Path(exists=True, path_type=Path), default=None,
              help="A .npy from the bf16 run, to score fidelity against.")
@click.option("--offload/--no-offload", default=True,
              help="Stream the backbone from host RAM. Off when the card fits it.")
@click.option("--out-dir", type=click.Path(path_type=Path), default=config.OUTPUT_DIR)
def benchmark(image, backend, gguf_path, resolution, runs, warmup, reference, offload, out_dir):
    """Time one configuration and, given a reference, score how far it drifted."""
    from marigoldedge.core.runner import MarigoldRunner

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, _ = imageio.load_rgb(image, resolution)
    runner = MarigoldRunner.build(backend, gguf_path, offload=offload)

    timing = runner.benchmark(rgb, runs=runs, warmup=warmup)
    depth = imageio.depth_to_array(runner.predict(rgb))

    record = {
        "backend": backend,
        "label": runner.label,
        "image": image.name,
        "resolution": resolution,
        "runs": runs,
        "warmup": warmup,
        "offload": offload,
        "seconds_median": timing.median,
        "seconds_all": timing.seconds,
        "peak_vram_gb": timing.peak_vram_gb,
    }
    if reference is not None:
        record.update(metrics.compare_to_reference(depth, np.load(reference)))

    # Offload state changes the numbers by a factor of three, so it is part of
    # the configuration and has to be part of the filename. Leaving it out let
    # the resident run silently overwrite the offloaded one.
    suffix = "" if offload else "_resident"
    stem = f"{image.stem}_{runner.label}_{resolution}{suffix}"
    np.save(out_dir / f"{stem}.npy", depth)
    imageio.colorize(depth).save(out_dir / f"{stem}.png")
    (out_dir / f"bench_{stem}.json").write_text(json.dumps(record, indent=2))
    logger.info("wrote bench_{}.json", stem)


@cli.command()
def fetch() -> None:
    """Download the Marigold checkpoint and its prompt embeddings (about 1.9 GB).

    This is everything `inspect` needs and nothing else: the 20.4 B backbone is
    a separate 40.9 GB download that only the benchmark requires. Keeping the
    two apart is what lets a reader check the post's central claim -- that the
    release is a LoRA over a frozen backbone -- without a GPU and without
    fetching the backbone it is a LoRA over.
    """
    from huggingface_hub import snapshot_download

    config.MARIGOLD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("downloading {} to {}", MARIGOLD_REPO, config.MARIGOLD_DIR)
    snapshot_download(
        MARIGOLD_REPO,
        local_dir=config.MARIGOLD_DIR,
        allow_patterns=[
            "depth/Log-stage2/trainables.safetensors",
            f"qwen_text_embeddings/{config.EMBED_PREFIX}_prompt_*.pt",
            "manifest.json",
        ],
        max_workers=4,
    )
    logger.info("done; now run: marigold-edge inspect")


@cli.command()
@click.option("--out-dir", type=click.Path(path_type=Path), default=config.OUTPUT_DIR)
def inspect(out_dir: Path):
    """Print what is inside the Marigold checkpoint, without loading a backbone.

    This is the claim the whole project rests on -- that the release is a LoRA
    over a frozen backbone -- so it is worth being able to check it in one
    command that needs no GPU.
    """
    from marigoldedge.core import weights

    lora, vae_decoder, rank, training_only = weights.split_trainables(config.TRAINABLES)
    targets = sorted({k.split(".lora_")[0].split(".")[-1] for k in lora if ".lora_" in k})

    def millions(d):
        return round(sum(t.numel() for t in d.values()) / 1e6, 1)

    total = millions(lora) + millions(vae_decoder) + millions(training_only)
    summary = {
        "lora_tensors": len(lora),
        "lora_rank": rank,
        "lora_target_modules": targets,
        "lora_params_millions": millions(lora),
        "vae_decoder_tensors": len(vae_decoder),
        "vae_params_millions": millions(vae_decoder),
        "training_only_tensors": len(training_only),
        "training_only_params_millions": millions(training_only),
        "total_tensors": len(lora) + len(vae_decoder) + len(training_only),
        "total_params_millions": round(total, 1),
        "lora_share_of_params": round(millions(lora) / total, 4),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoint_anatomy.json").write_text(json.dumps(summary, indent=2))
    click.echo(json.dumps(summary, indent=2))
