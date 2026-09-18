#!/usr/bin/env bash
# The whole pod session: environment, weights, four measurements, fidelity.
#
# The question this answers: on a card that holds every variant without
# streaming any of it, does quantizing the backbone cost time or save it? The
# laptop cannot answer that, because there every variant is bound by PCIe
# rather than by the GPU.
#
# Launch with:
#   setsid nohup /workspace/marigold-v2-edge/scripts/pod_run.sh \
#       > /workspace/run.log 2>&1 < /dev/null & disown
set -u

cd /workspace/marigold-v2-edge || exit 1
export MARIGOLD_EDGE_MODELS_DIR=/workspace/models
export MARIGOLD_EDGE_OUTPUT_DIR=/workspace/out
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
mkdir -p "$MARIGOLD_EDGE_OUTPUT_DIR"

IMAGE=data/samples/15_kitten.jpg
RES=768
RUNS=5
WARMUP=2

echo "=== STEP 1: environment $(date -Is)"
# The image already ships torch built for this driver. Installing our own would
# swap it for a wheel resolved against the laptop's CUDA, so torch is excluded
# here and the lock file is deliberately not used.
pip install -q uv 2>&1 | tail -2
uv pip install --system -q \
    "diffusers>=0.40" "transformers>=4.50" "accelerate>=1.0" "peft>=0.14" \
    "safetensors>=0.5" "gguf>=0.13" "bitsandbytes>=0.45" "huggingface-hub>=0.30" \
    "hf-transfer" "numpy>=1.26" "pillow>=10" "matplotlib>=3.9" "pandas>=2.2" \
    "click>=8.1" "loguru>=0.7" "tabulate>=0.9" "tqdm>=4.66" 2>&1 | tail -3
uv pip install --system -q --no-deps -e . 2>&1 | tail -2
python -c "import torch, diffusers, peft; print('torch', torch.__version__, 'diffusers', diffusers.__version__, 'peft', peft.__version__)"

echo "=== STEP 2: weights $(date -Is)"
python scripts/pod_fetch.py || { echo "Error: fetch failed"; exit 1; }

echo "=== STEP 3: bf16 reference $(date -Is)"
marigold-edge benchmark --image "$IMAGE" --backend bf16 --no-offload \
    --resolution "$RES" --runs "$RUNS" --warmup "$WARMUP" \
    || { echo "Error: bf16 row failed"; exit 1; }

REF="$MARIGOLD_EDGE_OUTPUT_DIR/15_kitten_bf16_${RES}.npy"
ls -la "$REF" || { echo "Error: no reference produced"; exit 1; }

echo "=== STEP 4: bitsandbytes NF4 $(date -Is)"
marigold-edge benchmark --image "$IMAGE" --backend nf4 \
    --resolution "$RES" --runs "$RUNS" --warmup "$WARMUP" --reference "$REF" \
    || echo "Error: nf4 row failed (continuing)"

echo "=== STEP 5: GGUF Q4_K_M, resident $(date -Is)"
marigold-edge benchmark --image "$IMAGE" --backend gguf --no-offload \
    --gguf-path "$MARIGOLD_EDGE_MODELS_DIR/gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf" \
    --resolution "$RES" --runs "$RUNS" --warmup "$WARMUP" --reference "$REF" \
    || echo "Error: gguf row failed (continuing)"

# The same file the laptop streamed, now streamed on a card that did not need
# to. It isolates what offloading costs from what the card costs, which is the
# only honest way to put the laptop's number in the same table as these.
echo "=== STEP 6: GGUF Q4_K_M, offloaded $(date -Is)"
marigold-edge benchmark --image "$IMAGE" --backend gguf --offload \
    --gguf-path "$MARIGOLD_EDGE_MODELS_DIR/gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf" \
    --resolution "$RES" --runs "$RUNS" --warmup "$WARMUP" --reference "$REF" \
    || echo "Error: gguf offloaded row failed (continuing)"

echo "=== STEP 7: summary $(date -Is)"
python - <<'PY'
import json, pathlib
out = pathlib.Path("/workspace/out")
rows = []
for f in sorted(out.glob("bench_*.json")):
    r = json.load(f.open())
    rows.append(r)
    print(f"{r['label']:<10} offload={str(r.get('offload')):<5} "
          f"{r['seconds_median']:>7.2f} s  {r['peak_vram_gb']:>6.2f} GB  "
          f"absrel_vs_ref={r.get('abs_rel_vs_ref', float('nan')):.5f}")
json.dump(rows, (out / "summary.json").open("w"), indent=2)
PY

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$MARIGOLD_EDGE_OUTPUT_DIR/hardware.txt"
python -c "import torch;print('torch', torch.__version__)" >> "$MARIGOLD_EDGE_OUTPUT_DIR/hardware.txt"
cat "$MARIGOLD_EDGE_OUTPUT_DIR/hardware.txt"

echo "=== DONE $(date -Is)"
