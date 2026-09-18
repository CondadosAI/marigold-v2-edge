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
# The image already ships torch built for this driver, and the base image is a
# PEP 668 externally-managed environment that refuses installs into it. A venv
# with --system-site-packages solves both at once: our packages are ours, and
# torch is inherited rather than resolved again against the laptop's CUDA.
# uv.lock is deliberately not used here for the same reason.
# Two traps here, both of which produce a working-looking environment that
# cannot see the GPU. The venv must use the *image's* interpreter, or
# --system-site-packages inherits nothing (uv defaults to its own Python and
# the image's torch lives in 3.12's dist-packages). And torch must be
# constrained to the version already installed, or uv resolves a fresh wheel
# built for a newer CUDA than the host driver: it imports, reports a version,
# and then says "driver too old" at the first CUDA call.
SYS_PY=$(which python)
echo "image torch: $($SYS_PY -c 'import torch;print(torch.__version__)') on $SYS_PY"

uv venv --python "$SYS_PY" --system-site-packages /workspace/venv 2>&1 | tail -2
VENV=/workspace/venv/bin
uv pip install --python $VENV/python -q \
    "diffusers>=0.40" "transformers>=4.50" "accelerate>=1.0" "peft>=0.14" \
    "safetensors>=0.5" "gguf>=0.13" "bitsandbytes>=0.45" "huggingface-hub>=0.30" \
    "hf-transfer" "numpy>=1.26" "pillow>=10" "matplotlib>=3.9" "pandas>=2.2" \
    "click>=8.1" "loguru>=0.7" "tabulate>=0.9" "tqdm>=4.66" 2>&1 | tail -3
uv pip install --python $VENV/python -q --no-deps -e . 2>&1 | tail -2

# uv resolves torch fresh from the default index, which ships a CUDA 13 build
# while this host's driver is 12.8. Pinning the version does not help: the
# version is the same, the CUDA build is not. Deleting torch from the venv is
# what makes --system-site-packages do its job, and the nvidia-* wheels go with
# it or they shadow the image's matching set.
uv pip uninstall --python $VENV/python torch torchvision triton 2>&1 | tail -1
$VENV/python - <<'PURGE'
import shutil, sys
from pathlib import Path
site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
removed = []
for child in site.iterdir():
    if child.name.startswith(("nvidia", "torch", "triton")):
        shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
        removed.append(child.name)
print("purged from venv:", len(removed))
PURGE

$VENV/python -c "
import torch, diffusers, peft
print('torch', torch.__version__, 'diffusers', diffusers.__version__, 'peft', peft.__version__)
assert torch.cuda.is_available(), 'torch cannot see the GPU -- wrong wheel for this driver'
print('cuda ok:', torch.cuda.get_device_name(0))
" || { echo "Error: environment incomplete"; exit 1; }

echo "=== STEP 2: weights $(date -Is)"
$VENV/python scripts/pod_fetch.py || { echo "Error: fetch failed"; exit 1; }

echo "=== STEP 3: bf16 reference $(date -Is)"
$VENV/marigold-edge benchmark --image "$IMAGE" --backend bf16 --no-offload \
    --resolution "$RES" --runs "$RUNS" --warmup "$WARMUP" \
    || { echo "Error: bf16 row failed"; exit 1; }

REF="$MARIGOLD_EDGE_OUTPUT_DIR/15_kitten_bf16_${RES}.npy"
ls -la "$REF" || { echo "Error: no reference produced"; exit 1; }

echo "=== STEP 4: bitsandbytes NF4 $(date -Is)"
$VENV/marigold-edge benchmark --image "$IMAGE" --backend nf4 \
    --resolution "$RES" --runs "$RUNS" --warmup "$WARMUP" --reference "$REF" \
    || echo "Error: nf4 row failed (continuing)"

echo "=== STEP 5: GGUF Q4_K_M, resident $(date -Is)"
$VENV/marigold-edge benchmark --image "$IMAGE" --backend gguf --no-offload \
    --gguf-path "$MARIGOLD_EDGE_MODELS_DIR/gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf" \
    --resolution "$RES" --runs "$RUNS" --warmup "$WARMUP" --reference "$REF" \
    || echo "Error: gguf row failed (continuing)"

# The same file the laptop streamed, now streamed on a card that did not need
# to. It isolates what offloading costs from what the card costs, which is the
# only honest way to put the laptop's number in the same table as these.
echo "=== STEP 6: GGUF Q4_K_M, offloaded $(date -Is)"
$VENV/marigold-edge benchmark --image "$IMAGE" --backend gguf --offload \
    --gguf-path "$MARIGOLD_EDGE_MODELS_DIR/gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf" \
    --resolution "$RES" --runs "$RUNS" --warmup "$WARMUP" --reference "$REF" \
    || echo "Error: gguf offloaded row failed (continuing)"

echo "=== STEP 7: summary $(date -Is)"
$VENV/python - <<'PY'
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
$VENV/python -c "import torch;print('torch', torch.__version__)" >> "$MARIGOLD_EDGE_OUTPUT_DIR/hardware.txt"
cat "$MARIGOLD_EDGE_OUTPUT_DIR/hardware.txt"

echo "=== DONE $(date -Is)"
