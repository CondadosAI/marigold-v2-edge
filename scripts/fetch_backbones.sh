#!/usr/bin/env bash
# Fetch the remaining backbones, one at a time so they do not fight for bandwidth.
#
# Four GGUF levels span the useful range (7.2 GB to 21.8 GB) and the bf16
# transformer serves two table rows at once: it is the unquantized reference,
# and bitsandbytes quantizes NF4 from it on load.
set -u

P=/media/lcondados/workspace/personal/marigold-v2-edge
GGUF_REPO=https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/resolve/main
cd "$P" || exit 1

wait_for_q4() {
  local f="$P/models/gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf"
  while [ ! -f "$f" ] || [ "$(stat -c%s "$f" 2>/dev/null || echo 0)" -lt 13000000000 ]; do
    sleep 60
  done
}

echo "waiting for the in-flight Q4_K_M download..."
wait_for_q4
echo "Q4_K_M complete at $(date -Is)"

for level in Q2_K Q6_K Q8_0; do
  out="models/gguf/Qwen-Image-Edit-2509-${level}.gguf"
  echo "--- ${level} -> ${out} at $(date -Is)"
  curl -L --retry 5 --retry-delay 10 -C - -s \
       -o "$out" "${GGUF_REPO}/Qwen-Image-Edit-2509-${level}.gguf"
  echo "    done: $(stat -c%s "$out" 2>/dev/null || echo FAILED) bytes"
done

echo "--- bf16 transformer (41 GB) at $(date -Is)"
HF_HOME="$P/models/hf" "$P/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
print('bf16 transformer at:', snapshot_download(
    'Qwen/Qwen-Image-Edit-2509', allow_patterns=['transformer/*'], max_workers=4))
"
echo "all backbones fetched at $(date -Is)"
