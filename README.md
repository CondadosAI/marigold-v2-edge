# Marigold V2 on 6 GB

Marigold V2 is published as a ~17 GB inference job. This repository runs it on a
6 GB laptop GPU, and measures what that costs.

The trick is not a trick. The released checkpoint is **not** a 20 B model: it is
a rank-128 LoRA over a *frozen* Qwen-Image-Edit-2509 backbone, plus a fine-tuned
VAE decoder. A frozen backbone is a swappable backbone, so the same 1.85 GB
checkpoint rides on a community GGUF quantization of that backbone and the
memory budget drops to whatever the quantization costs.

Verify that claim yourself, without a GPU and without downloading a backbone:

```bash
uv sync
uv run marigold-edge inspect
```

## Layout

```
src/marigoldedge/
├── config.py            # project-root-aware paths, env-overridable
├── cli.py               # predict / benchmark / inspect
└── core/
    ├── weights.py       # split trainables.safetensors into LoRA / VAE / training-only
    ├── backends.py      # load the backbone as NF4, GGUF or bf16; attach the LoRA
    ├── pipeline.py      # the single-step depth graph
    ├── imageio.py       # image in, depth array and colour preview out
    ├── metrics.py       # affine-aligned AbsRel, delta1, RMSE
    └── runner.py        # assemble a backend and time it
output/                  # saved artifacts; every number in the article comes from here
```

## Running it

```bash
# one depth map from a GGUF backbone
uv run marigold-edge predict --image data/samples/15_kitten.jpg \
    --backend gguf --gguf-path models/gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf

# time it, and score the drift against a reference run
uv run marigold-edge benchmark --image data/samples/15_kitten.jpg \
    --backend gguf --gguf-path models/gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf \
    --resolution 768 --runs 5 --warmup 2 \
    --reference output/15_kitten_bnb-NF4_768.npy
```

## What this is not

- **Not a TensorRT or OpenVINO comparison.** Both were ruled out before any code
  was written, for a specific reason: bitsandbytes NF4 and SVDQuant are
  CUDA-kernel-only, and no exporter for a 20 B MMDiT exists in optimum-intel or
  TensorRT. Neither runtime can take this model today.
- **Not a Nunchaku comparison.** Nunchaku's SVDQuant INT4 would be the fastest
  path, but it cannot load a custom Qwen-Image LoRA: `nunchaku/lora/` implements
  FLUX only, and its Lightning variants are fused at quantization time. Reaching
  INT4 would mean fusing Marigold's LoRA into bf16 weights and re-running
  SVDQuant calibration on a 20 B model.
- **Not a new model.** All the accuracy here is Marigold V2's. This repository
  only changes where the backbone's weights live.

## Licences — read this before you ship anything built on it

The Apache-2.0 `LICENSE` at the root covers **the code in `src/`**. It does not
cover the weights this code downloads at run time. Each of those carries its
own terms:

| Artifact | Source | Licence |
|---|---|---|
| Marigold V2 checkpoints | `huawei-bayerlab/marigold-v2-0` | Apache-2.0 |
| Qwen-Image-Edit-2509 backbone | `Qwen/Qwen-Image-Edit-2509` | Apache-2.0 |
| GGUF quantizations | `QuantStack/Qwen-Image-Edit-2509-GGUF` | *see repo — verify before commercial use* |

This repository **downloads** these; it does not redistribute them.

## Citation

Pavlovic, I., Wandel, T., Obukhov, A., Bartolomei, L., Davydov, A., Tosi, F.,
Poggi, M., Süsstrunk, S., Dai, D. *Marigold V2: Revisiting Diffusion
Transformers for Monocular Depth Estimation*. SIGGRAPH Asia 2026.
<https://doi.org/10.1145/3842528> · <https://arxiv.org/abs/2609.08084>

The inference graph in `core/pipeline.py` is transcribed from the authors'
reference implementation and cross-checked against their training repository.
