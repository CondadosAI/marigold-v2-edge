#!/usr/bin/env bash
# Second session: three more quantizers, on the same A40 as the first, so the
# rows are comparable to the ones already published.
#
# The point of torchao-int4 is not another data point. The article's finding is
# that which 4-bit you pick costs more than the decision to quantize, and that
# rested on two samples. This is the third.
#
# float8 is deliberately absent: torchao's fp8 schemes want compute capability
# 8.9 and an A40 is 8.6. Measuring it means changing the GPU, which would make
# every published row incomparable.
set -u

cd /workspace/marigold-v2-edge || exit 1
export MARIGOLD_EDGE_MODELS_DIR=/workspace/models
export MARIGOLD_EDGE_OUTPUT_DIR=/workspace/out
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1
mkdir -p "$MARIGOLD_EDGE_OUTPUT_DIR"

IMAGE=data/samples/15_kitten.jpg
RES=768; RUNS=5; WARMUP=2
VENV=/workspace/venv/bin

echo "=== STEP 1: environment $(date -Is)"
SYS_PY=$(which python)
uv venv --python "$SYS_PY" --system-site-packages /workspace/venv 2>&1 | tail -1
uv pip install --python $VENV/python -q \
    "diffusers>=0.40" "transformers>=4.50" "accelerate>=1.0" "peft>=0.14" \
    "safetensors>=0.5" "gguf>=0.13" "bitsandbytes>=0.45" "huggingface-hub>=0.30" \
    "torchao>=0.9" "numpy>=1.26" "pillow>=10" "matplotlib>=3.9" "pandas>=2.2" \
    "click>=8.1" "loguru>=0.7" "tabulate>=0.9" "tqdm>=4.66" 2>&1 | tail -3
uv pip install --python $VENV/python -q --no-deps -e . 2>&1 | tail -1
# uv resolves torch from the default index, which ships a CUDA build newer than
# this host's driver. Deleting it is what makes --system-site-packages work.
uv pip uninstall --python $VENV/python torch torchvision triton 2>&1 | tail -1
$VENV/python - <<'PURGE'
import shutil, sys
from pathlib import Path
site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
for child in site.iterdir():
    if child.name.startswith(("nvidia", "torch", "triton")) and not child.name.startswith("torchao"):
        shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
print("purged")
PURGE
$VENV/python - <<'VERS' | tee "$MARIGOLD_EDGE_OUTPUT_DIR/hardware2.txt"
import importlib.metadata as md, sys, torch
print("python", sys.version.split()[0])
assert torch.cuda.is_available(), "torch cannot see the GPU"
cap = torch.cuda.get_device_capability(0)
print("gpu", torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}")
for pkg in ("torch","diffusers","peft","transformers","bitsandbytes","torchao","accelerate","gguf","numpy"):
    try: print(pkg, md.version(pkg))
    except md.PackageNotFoundError: print(pkg, "NOT INSTALLED")
VERS
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "Error: environment incomplete"; exit 1; }

echo "=== STEP 2: weights $(date -Is)"
$VENV/python scripts/pod_fetch.py || { echo "Error: fetch failed"; exit 1; }
REF="$MARIGOLD_EDGE_OUTPUT_DIR/15_kitten_bf16_768.npy"

echo "=== STEP 3: bf16 reference (re-measured so this session is self-contained) $(date -Is)"
$VENV/marigold-edge benchmark --image "$IMAGE" --backend bf16 --no-offload \
    --resolution "$RES" --runs "$RUNS" --warmup "$WARMUP" || { echo "Error: bf16 failed"; exit 1; }

for B in torchao-int8 torchao-int4 bnb-int8; do
  echo "=== $B $(date -Is)"
  $VENV/marigold-edge benchmark --image "$IMAGE" --backend "$B" \
      --resolution "$RES" --runs "$RUNS" --warmup "$WARMUP" --reference "$REF" \
      || echo "Error: $B row failed (continuing)"
done

echo "=== SUMMARY $(date -Is)"
$VENV/python - <<'PY'
import json, pathlib
for f in sorted(pathlib.Path("/workspace/out").glob("bench_*.json")):
    r = json.load(f.open())
    print(f"{r['label']:<14} {r['seconds_median']:>6.2f}s  {r['peak_vram_gb']:>6.2f}GB  "
          f"rmse={r.get('rmse_vs_ref','-')}")
PY
echo "=== DONE $(date -Is)"
