"""Load the Qwen-Image-Edit-2509 MMDiT at a chosen precision, then fit Marigold's LoRA.

Every backend returns the same object -- a ``QwenImageTransformer2DModel`` with
the adapter attached -- so the benchmark varies the backbone and nothing else.
What differs is the format the 20.4 B parameters are stored in:

* ``nf4``   — bitsandbytes 4-bit, what upstream ships. CUDA-kernel only, which
  is exactly why it cannot be exported to ONNX, OpenVINO or TensorRT.
* ``gguf``  — a community quantization, dequantized per layer on the fly. This
  is the one this project measures.
* ``bf16``  — the unquantized reference, ~41 GB, included for completeness.

**Nothing here fits a 6 GB card outright.** The smallest published GGUF of this
backbone is Q2_K at 7.15 GB and the card has 5.67 GiB usable, so every
configuration streams weights from host RAM through
``apply_group_offloading``. That is not a fallback for the low-VRAM case; it is
the only case. It also means latency is bounded by PCIe bandwidth rather than
by the GPU, which is the single most important thing to know before reading any
number this repository produces.

The NF4 path mirrors ``marigoldv2_inference.py`` from the authors' Hugging Face
Space call for call, including the skipped module and the fp32 preparation
step, so that its output is the correctness reference the other backends are
scored against rather than a near-enough reimplementation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import QwenImageTransformer2DModel
from diffusers.hooks import apply_group_offloading
from loguru import logger
from peft import LoraConfig, prepare_model_for_kbit_training

QWEN_REPO = "Qwen/Qwen-Image-Edit-2509"
COMPUTE_DTYPE = torch.bfloat16

# Transcribed from the authors' reference implementation. Note `txt_in`, which
# the depth checkpoint does not actually carry -- the adapter is built to the
# recipe's shape and the checkpoint fills in the subset it trained.
LORA_TARGET_MODULES = [
    "img_in",
    "txt_in",
    "norm_out.linear",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]
LORA_RANK = 128
LORA_ALPHA = 128
# bitsandbytes must leave this one in higher precision or the first block's
# modulation collapses; upstream skips it explicitly.
NF4_SKIP_MODULES = ["transformer_blocks.0.img_mod"]

# How many of the 60 transformer blocks are resident on the GPU at once. Lower
# means less VRAM and more host-to-device traffic; this is the dial the
# benchmark turns when a level will not fit.
DEFAULT_BLOCKS_PER_GROUP = 1


def offload(
    transformer,
    device: str,
    blocks_per_group: int = DEFAULT_BLOCKS_PER_GROUP,
    use_stream: bool = False,
) -> object:
    """Stream the model from host RAM one block group at a time.

    Call this **after** the LoRA weights are loaded. Offloading installs hooks
    that move parameters between host and device; weights written into the
    model afterwards are not tracked by those hooks and silently stay behind.

    ``use_stream`` overlaps the next group's copy with the current group's
    compute, which is the difference between being PCIe-bound once and twice.
    It is off by default because it requires *pinned* host memory: pinning a
    13 GB backbone needs another 13 GB that cannot be swapped, and on a 31 GB
    machine that is what the kernel's OOM killer reaches for. Turn it on when
    host RAM has room, and expect it to be the single biggest latency win.

    ``low_cpu_mem_usage`` keeps the CPU-side copy lean at a small cost per
    onload, which is the right trade when the host is the constrained side --
    and with a 20.4 B backbone on a laptop, it is.
    """
    apply_group_offloading(
        transformer,
        onload_device=torch.device(device),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=blocks_per_group,
        use_stream=use_stream,
        non_blocking=use_stream,
        low_cpu_mem_usage=True,
    )
    logger.info(
        "group offloading: {} block(s) resident, stream={}",
        blocks_per_group, use_stream,
    )
    return transformer


def _attach_adapter(transformer, kbit: bool):
    """Create the empty LoRA the checkpoint will be loaded into.

    ``prepare_model_for_kbit_training`` is not about training here. Training
    kept the non-quantized parameters in fp32 through this call, so mirroring
    it is what makes the numerics match upstream rather than merely resemble
    it.
    """
    if kbit:
        prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    transformer.add_adapter(
        LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES)
    )
    for name, param in transformer.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(COMPUTE_DTYPE)
    transformer.requires_grad_(False)
    return transformer


def load_nf4(device: str = "cuda"):
    """bitsandbytes NF4 — the upstream default, and the correctness reference."""
    from diffusers import BitsAndBytesConfig

    logger.info("loading NF4 backbone from {}", QWEN_REPO)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        QWEN_REPO,
        subfolder="transformer",
        torch_dtype=COMPUTE_DTYPE,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            llm_int8_skip_modules=NF4_SKIP_MODULES,
        ),
        device_map=device,
    )
    return _attach_adapter(transformer, kbit=True)


def load_gguf(gguf_path: Path):
    """A GGUF-quantized backbone, dequantized per layer as it runs."""
    from diffusers import GGUFQuantizationConfig

    logger.info(
        "loading GGUF backbone: {} ({:.1f} GB on disk)",
        gguf_path.name,
        gguf_path.stat().st_size / 1e9,
    )
    transformer = QwenImageTransformer2DModel.from_single_file(
        str(gguf_path),
        quantization_config=GGUFQuantizationConfig(compute_dtype=COMPUTE_DTYPE),
        config=QWEN_REPO,
        subfolder="transformer",
        torch_dtype=COMPUTE_DTYPE,
    )
    return _attach_adapter(transformer, kbit=False)


def load_bf16():
    """The unquantized reference. ~41 GB, so offload does all the work."""
    logger.warning("bf16 backbone is ~41 GB; every block is streamed from host RAM")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        QWEN_REPO, subfolder="transformer", torch_dtype=COMPUTE_DTYPE
    )
    return _attach_adapter(transformer, kbit=False)


def load_lora_(transformer, lora_state: dict[str, torch.Tensor]) -> None:
    """Load Marigold's weights into the adapter created above, in place.

    Loaded non-strictly because the model also holds the 20.4 B frozen
    parameters the checkpoint says nothing about. A *missing LoRA* key is fatal
    though: it would mean part of the adapter stayed at its random
    initialisation, which produces a depth map that looks plausible and is
    wrong. That is the failure worth crashing on.
    """
    missing, unexpected = transformer.load_state_dict(lora_state, strict=False)
    missing_lora = [k for k in missing if "lora_" in k]
    if unexpected or missing_lora:
        raise ValueError(
            f"LoRA did not land cleanly: {len(unexpected)} unexpected "
            f"(e.g. {unexpected[:3]}), {len(missing_lora)} missing "
            f"(e.g. {missing_lora[:3]})"
        )
    logger.info("Marigold LoRA loaded: {} tensors, no key mismatches", len(lora_state))
