# What quantizing Marigold V2 actually costs

Marigold V2 is published as a ~17 GB inference job. This repository swaps the
backbone underneath it for a quantized one and measures, on a single rented
A40, exactly what that buys and what it costs.

The trick is not a trick. The released checkpoint is **not** a 20 B model: it is
a rank-128 LoRA over a *frozen* Qwen-Image-Edit-2509 backbone, plus a fine-tuned
VAE decoder. A frozen backbone is a swappable backbone, so the same 1.85 GB
checkpoint rides on a community GGUF quantization of that backbone and the
memory budget drops to whatever the quantization costs.

Measured at 768 px on one image, bf16 on the A40 as the reference:

| Backend | Placement | s | Peak VRAM | rmse/range | Pearson r |
|---|---|---:|---:|---:|---:|
| bf16 | resident | 0.79 | 44.72 GB | reference | — |
| bnb-NF4 | resident | 0.87 | 15.44 GB | 1.92% | 0.9972 |
| GGUF Q4_K_M | resident | 1.62 | 17.04 GB | 1.56% | 0.9982 |
| GGUF Q4_K_M | offloaded | 5.40 | 2.15 GB | 1.56% | 0.9982 |

Going 4-bit costs 10% of the time and saves 65% of the memory. Picking GGUF
over NF4 costs another 1.87x and saves nothing. Streaming from host RAM costs
3.33x. The whole table is one A40 hour, about fifty cents.

Verify that claim yourself, without a GPU and without downloading the backbone.
The checkpoint is 1.9 GB; the backbone it rides on is a separate 40.9 GB that
only the benchmark needs.

```bash
uv sync
uv run marigold-edge fetch     # the Marigold checkpoint only, ~1.9 GB
uv run marigold-edge inspect   # prints the anatomy, no GPU required
```

That prints the split the post opens with: 1,446 LoRA tensors at rank 128,
108 VAE-decoder tensors, and 2 training-only tensors, 926.0 M parameters in all.

## Layout

```
tests/                   # uv run pytest — no GPU, no downloads, no network
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

To regenerate the article's table from the artifacts committed under `output/`,
with no GPU and no downloads at all:

```bash
uv run python scripts/build_table.py
```

The whole benchmark session, on a fresh GPU host, is one script:

```bash
scripts/pod_run.sh          # environment, weights, four rows, summary
```

Or a single configuration by hand:

```bash
uv run marigold-edge predict --image data/samples/15_kitten.jpg \
    --backend gguf --no-offload \
    --gguf-path models/gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf

uv run marigold-edge benchmark --image data/samples/15_kitten.jpg \
    --backend gguf --no-offload \
    --gguf-path models/gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf \
    --resolution 768 --runs 5 --warmup 2 \
    --reference output/a40/15_kitten_bf16_768.npy
```

`--offload` streams the backbone from host RAM one block at a time, for a card
that cannot hold it. It needs roughly the backbone's size free in host RAM, and
running out of it is worse than slow: the OOM is preceded by swap thrash.

## What this is not

- **Not a TensorRT or OpenVINO comparison.** bitsandbytes NF4 and SVDQuant are
  CUDA-kernel-only. OpenVINO is no longer out of reach -- optimum-intel now
  registers `qwenimage-transformer` and both VAE halves as exportable -- but
  reaching it means merging the LoRA into bf16 and exporting a 20 B model,
  which is its own piece of work and not this one.
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
