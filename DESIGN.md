# DESIGN — Marigold V2 on 6 GB

Phase-1 contract (new-article-project). Written 2026-09-12. This is the
agreement for the rest of the build; change it here before changing the code.

## The hook

Marigold V2 ships as a ~17 GB inference job: the paper reports 1.9 s at 1024²
on a single 32 GB GPU [cite, not reproduced]. But the released checkpoint is
**not** a 20 B model. It is a rank-128 PEFT LoRA (926 M params, 1.85 GB) over a
frozen Qwen-Image-Edit-2509 MMDiT, plus a fine-tuned VAE decoder.

A frozen backbone is a swappable backbone. Community GGUF quantizations of
Qwen-Image-Edit-2509 already exist (Q2_K 7.2 GB → Q8_0 21.8 GB). Put Marigold's
LoRA and VAE decoder on a quantized backbone and the model runs on a 6 GB
laptop GPU.

**The question the post answers:** what does that cost, in seconds per image and
in depth accuracy, at each quantization level — and where does it stop being
depth estimation and start being noise?

## Reader & takeaway

A practitioner who read the Marigold V2 announcement, wants to run it, and has
one consumer GPU. After the post they know which GGUF level to pick for their
VRAM budget, what accuracy they give up, and how to apply the same
LoRA-on-a-quantized-backbone trick to the next frozen-backbone model.

## What is actually being measured

Verified from source before committing (`trainables.safetensors` header,
`network_graph.py`, `inference_depth.yaml`):

- 1,446 of 1,556 tensors are PEFT LoRA (`lora_A`/`lora_B`, `.default.`), rank
  128, over 60 transformer blocks + `img_in` + `norm_out.linear`.
- `img_in.lora_A` is `[128, 64]` → 64 input channels, unchanged. No
  architecture surgery; it is a drop-in LoRA.
- The remainder is a fully fine-tuned VAE **decoder** and an
  `iREPAStudentProjector` that is training-only.
- Inference graph: VAE encode → **one** rectified-flow step
  (`num_inference_steps: 1`, `guidance_scale: 1.0`) → VAE decode →
  `SelectChannel mean`. The text encoder never loads: prompt embeddings are
  precomputed `.pt` files (`qwen_edit_2509_qwen_depth_realimg512`, 19 MB).

That last point is what makes this tractable — no 8.3 B Qwen2.5-VL text encoder
at runtime, and a single step means latency is one forward pass, not a schedule.

## Hardware (the constraint that defines the post)

| | |
|---|---|
| GPU | RTX 3060 Laptop, **6 GB** VRAM (Ampere → bf16 native, INT4-capable) |
| CPU | i7-12700H, 20 threads |
| RAM | 31 GB |
| Disk | 261 GB free on the workspace volume |

The paper's reference is a 32 GB GPU. We have 6 GB. That gap is the article.

## Benchmark axes

Cut down from the original sweep to the smallest set that answers the question
(Luis, 2026-09-18: simplest first). A GGUF level sweep and a resolution sweep
are post #3, not this one.

- **On the A40 (48 GB, everything resident)**: bf16 · bnb-NF4 · GGUF Q4_K_M,
  plus GGUF Q4_K_M *offloaded* on a card that did not need to, which separates
  what offloading costs from what the backbone costs.
- **On the laptop (6 GB)**: GGUF Q4_K_M only. bf16 and NF4 are not slow there,
  they are impossible, and saying why is a row of the table.
- **Resolution**: 768 px long edge, everywhere.
- **Reported per cell**: seconds/image (median of N after warmup), peak VRAM,
  peak host RAM, on-disk size.
- **Fidelity**: against the bf16 reference run on the same images, after affine
  alignment — AbsRel and δ1. Plus GT metrics if the repo's `evaluation/`
  configs make NYUv2 or ETH3D cheap to wire.

Fidelity-vs-reference is the primary honesty metric: it isolates what
quantization costs, independent of what Marigold gets wrong to begin with.

## Runtime decisions (settled, do not relitigate)

- **GGUF is the quantized runtime.** diffusers 0.38 `GGUFQuantizationConfig`,
  LoRA loaded on top.
- **Nunchaku is a documented wall.** Verified in source: `nunchaku/lora/`
  contains only `flux`, and `transformer_qwenimage.py` has no
  `update_lora_params`. Custom LoRA on Qwen-Image is unsupported; their
  Lightning models are fused at quantization time. Reaching INT4 would mean
  fuse-then-requantize with deepcompressor on a 20 B model. Out of scope,
  stated in Limitations, and a candidate for a follow-up post.
- **No TensorRT, and no Innovator tags on this post.** bitsandbytes NF4 and
  SVDQuant are CUDA-kernel-only. A "why not" paragraph does not earn the
  `openvino`/`IntelSoftwareInnovator` tags (Luis, 2026-09-12), and nothing here
  measures OpenVINO.
- **OpenVINO is no longer a wall — it is post #2 (checked 2026-09-18).**
  optimum-intel now registers `qwenimage-transformer`, `qwenimage-vae-encoder`
  and `qwenimage-vae-decoder` as exportable, with `OVModelQwenImageTransformer`
  and matching model patchers. There is no `OVQwenImageEditPipeline`, but
  Marigold does not use the Edit *pipeline*: it drives the transformer and VAE
  directly, exactly as `core/pipeline.py` does, and the transformer is the same
  `QwenImageTransformer2DModel` class the config covers. The route is merge the
  LoRA into bf16, export transformer + VAE to IR, run on an Intel Max via Intel
  Tiber AI Cloud. That is where the Innovator tag gets earned with a measured
  number, and it inherits this post's bf16 reference to compare against.
- **The reference host is a RunPod A40, not Kaggle.** Kaggle's free tiers are
  T4/P100 (Turing/Pascal, no native bf16) and the config is bf16, so they
  cannot host a faithful reference. An A40 is Ampere, holds all three variants
  resident, and costs about $0.50 for the session.

## Running scenario

One named real image carries every worked example in the post (CLAUDE.md: when a
post measures something, the scenario *is* the thing measured). Candidate: a
scene with the fur/foliage/hair-thin detail the paper claims to resolve, so the
quantization damage is visible rather than argued. Pick it in Phase 2 from the
repo's own `assets/` sample set and name it in the post.

## Hero visual

A grid: same image, depth map at each quantization level, with AbsRel-vs-
reference under each panel — the point where the depth map visibly falls apart
should be findable by eye, then confirmed by the number.

Second chart: seconds/image and peak VRAM vs quantization level, with a 6 GB
line drawn across it.

## Risks / fallbacks

1. **LoRA-on-GGUF may not load cleanly.** Marigold's keys are prefixed
   `Diffuser.` and suffixed `.default`; they need renaming to diffusers'
   convention. Fallback: merge LoRA into bf16 weights on CPU, then quantize to
   GGUF ourselves with `gguf-py`.
2. **The VAE decoder swap.** Marigold's decoder is full-weight and must replace
   the stock Qwen VAE decoder. Should be straightforward (it is small) but is
   the most likely source of a silently wrong depth map. Guard: compare against
   the official pipeline on one image before trusting any table.
3. **The bf16 reference may be very slow** on 6 GB with offload — ~10 GB
   streamed over PCIe per step. If it is minutes/image, run the reference on a
   handful of images only and say so in the reproducibility block.
4. **Q2_K/Q3_K may simply not produce depth.** That is a finding, not a
   failure; it is the "where it stops working" half of the hook.

## Deliverables

- `CondadosAI/marigold-v2-edge` — public, Apache-2.0, uv `src/` layout, click
  CLI, saved `output/` artifacts, notebook with outputs cleared. README lists
  every third-party weight/dataset with its verified licence.
- The post on condados.ai (category: Edge Deployment), then `/medium-post` and
  `/linkedin-post` against the live URL.

## Licences to verify before the repo goes public

| Artifact | Stated | Verified? |
|---|---|---|
| Qwen-Image-Edit-2509 | Apache-2.0 | yes (HF model card) |
| Marigold V2 weights | Apache-2.0 | yes (repo + HF) |
| QuantStack GGUF quants | inherits base | **to check** |
| NYUv2 / ETH3D (if used) | non-commercial / unstated | **to check** — this is an Edge post, not Fundamentals, so state terms explicitly |
