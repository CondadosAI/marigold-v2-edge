#!/usr/bin/env bash
# Re-measure the laptop row under conditions we can state, while watching the
# thermals of a drive that has failed three times.
#
# Why this exists as a script rather than a command: the number it produces is
# the article's hook, and the first time it was taken nobody recorded whether
# the machine was on mains. A laptop GPU on battery is a different GPU, and a
# published latency that cannot say which one it used is not reproducible.
#
# The NVMe is read once, for the 13 GB backbone load. Offloading streams from
# host RAM after that, so the sustained phase is memory traffic and not disk --
# but the load itself is a 13 GB sequential read, which is exactly the access
# pattern that heats a laptop M.2.
set -u

cd "$(dirname "$0")/.." || exit 1
SAMPLES=/tmp/marigold_thermals.csv
: > "$SAMPLES"

sample_thermals() {
  echo "t,nvme0_c,nvme1_c,gpu_c,gpu_mhz,ram_avail_gb" >> "$SAMPLES"
  while true; do
    local temps=()
    for h in /sys/class/hwmon/hwmon*; do
      [ "$(cat "$h/name" 2>/dev/null)" = "nvme" ] && \
        temps+=("$(awk '{printf "%.0f", $1/1000}' "$h/temp1_input" 2>/dev/null)")
    done
    local gpu
    gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    local ram
    ram=$(free -g | awk '/^Mem:/{print $7}')
    echo "$(date +%s),${temps[0]:-},${temps[1]:-},${gpu},${ram}" >> "$SAMPLES"
    sleep 3
  done
}

NEED_RAM_GB=${NEED_RAM_GB:-19}
MAX_NVME_C=${MAX_NVME_C:-72}
START_NVME_C=${START_NVME_C:-65}

hottest_nvme() {
  local hot=0 t
  for h in /sys/class/hwmon/hwmon*; do
    if [ "$(cat "$h/name" 2>/dev/null)" = "nvme" ]; then
      t=$(awk '{printf "%.0f", $1/1000}' "$h/temp1_input" 2>/dev/null)
      [ -n "$t" ] && [ "$t" -gt "$hot" ] && hot=$t
    fi
  done
  echo "$hot"
}

# Two preflight gates, both learned the hard way on 18 Sep 2026. Running out of
# host RAM does not merely fail: the OOM kill is preceded by swap thrash that
# took this machine's swap NVMe from 61 C to 76 C, on a drive that has already
# died three times. The data disk holding the 13 GB model stayed under 68 C
# throughout, so the risk is memory pressure, not file size.
POWER=$(cat /sys/class/power_supply/AC*/online 2>/dev/null | head -1 | sed 's/1/on mains/;s/0/ON BATTERY/')
RAM_AVAIL=$(free -g | awk '/^Mem:/{print $7}')
NVME_NOW=$(hottest_nvme)
echo "=== power: $POWER"
echo "=== ram available: ${RAM_AVAIL} GB (need >= ${NEED_RAM_GB})"
echo "=== hottest nvme: ${NVME_NOW} C (need <= ${START_NVME_C})"

[ "$POWER" = "on mains" ] || { echo "REFUSING: on battery, the GPU clocks differently"; exit 2; }
[ "$RAM_AVAIL" -ge "$NEED_RAM_GB" ] || {
  echo "REFUSING: ${RAM_AVAIL} GB free, need ${NEED_RAM_GB}. Close applications first."
  echo "  A short run is not worth thrashing swap on this drive."
  exit 2
}
[ "$NVME_NOW" -le "$START_NVME_C" ] || {
  echo "REFUSING: nvme at ${NVME_NOW} C, let it cool to ${START_NVME_C} C first"; exit 2
}

sample_thermals & SAMPLER=$!
watch_heat() {
  while true; do
    if [ "$(hottest_nvme)" -gt "$MAX_NVME_C" ]; then
      echo "=== ABORTING: nvme exceeded ${MAX_NVME_C} C mid-run"
      pkill -f "[m]arigold-edge benchmark"
      return
    fi
    sleep 5
  done
}
watch_heat & WATCHER=$!
trap 'kill $SAMPLER $WATCHER 2>/dev/null' EXIT

echo "=== asserting the GPU is the one we think it is"
.venv/bin/python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA'
cap = torch.cuda.get_device_capability(0); arch = f'sm_{cap[0]}{cap[1]}'
assert arch in torch.cuda.get_arch_list(), f'{arch} not in {torch.cuda.get_arch_list()}'
print('   ', torch.cuda.get_device_name(0), arch, f'{torch.cuda.mem_get_info(0)[1]/1e9:.1f} GB')
" || exit 1

echo "=== benchmark"
.venv/bin/marigold-edge benchmark \
    --image data/samples/15_kitten.jpg --backend gguf --offload \
    --gguf-path models/gguf/Qwen-Image-Edit-2509-Q4_K_M.gguf \
    --resolution 768 --runs 5 --warmup 2 \
    --reference output/a40/15_kitten_bf16_768.npy
STATUS=$?

kill $SAMPLER $WATCHER 2>/dev/null
echo "=== thermals over the run"
awk -F, 'NR>1 && $2!="" {
  n0=$2; n1=$3; g=$4;
  if (n0>m0) m0=n0; if (n1>m1) m1=n1; if (g>mg) mg=g;
  if (mr=="" || $6<mr) mr=$6;
  c++
} END {
  printf "    samples=%d  nvme0 peak=%d C  nvme1 peak=%d C  gpu peak=%d C  ram low=%s GB\n", c, m0, m1, mg, mr
}' "$SAMPLES"
cp "$SAMPLES" output/laptop_thermals.csv
echo "=== exit $STATUS"
exit $STATUS
