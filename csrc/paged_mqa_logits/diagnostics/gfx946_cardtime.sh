#!/bin/bash
# gfx946 card-time runbook for the native FP4 MMAC path.
#
# Run this FIRST when a gfx946 part becomes available.  Everything that does
# not need the card has already been done and checked on gfx938, so this script
# is pure execution: it does not compile anything that was not already proven to
# compile, and it reports one verdict per stage.
#
#   bash gfx946_cardtime.sh            full run
#   bash gfx946_cardtime.sh quick      stages 1-2 only (~2 min)
#
# Stages
#   1  sanity        device really is gfx946
#   2  probe         native MMAC numerical test (settles the accumulator map)
#   3  extension     build the python extension with the native path + strict
#   4  differential  operator output, native vs the shipping path
#   5  benchmark     native vs shipping timing
set -u
MODE="${1:-full}"
REPO=/home/projects/lightop-repo/lightop-gfx946-opt
LOG=/home/gfx946_cardtime_$(date +%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"

say() { printf '\n========== %s ==========\n' "$1"; }
VERDICT=""
note() { VERDICT="$VERDICT
$1"; echo "$1"; }

# ---------------------------------------------------------------- stage 1
say "1  sanity"
ARCH=$(python3 -c 'import torch;print(torch.cuda.get_device_properties(0).gcnArchName)' 2>/dev/null)
echo "arch reported by torch: ${ARCH:-<none>}"
case "$ARCH" in
  gfx946*) note "stage1 PASS  device is $ARCH" ;;
  "")      note "stage1 FAIL  torch sees no device"; echo "$VERDICT"; exit 1 ;;
  *)       note "stage1 ABORT device is $ARCH, not gfx946 -- nothing below can run"
           echo "$VERDICT"; exit 1 ;;
esac
echo "CUs: $(python3 -c 'import torch;print(torch.cuda.get_device_properties(0).multi_processor_count)')"

# ---------------------------------------------------------------- stage 2
say "2  native MMAC probe"
cd "$REPO/csrc/paged_mqa_logits" || exit 1
hipcc -x hip --offload-arch=gfx946 -O2 diagnostics/fp4_mmac_probe.cu -o /tmp/probe || {
  note "stage2 FAIL  probe build broke"; echo "$VERDICT"; exit 1; }
if /tmp/probe; then
  note "stage2 PASS  native FP4 MMAC numerically verified"
else
  note "stage2 FAIL  see LAYOUT_MISMATCH / NATIVE_FP4_FAILED above"
  note "             -> fix the fragment or accumulator assumption first;"
  note "                stages 3-5 are meaningless until this passes"
  echo "$VERDICT"; exit 1
fi
[ "$MODE" = quick ] && { echo "$VERDICT"; exit 0; }

# ---------------------------------------------------------------- stage 3
say "3  extension build"
cd "$REPO" || exit 1
# Shipping build (native off) -- the reference for stage 4.
echo "--- building shipping extension (native OFF) ---"
PYTORCH_ROCM_ARCH=gfx946 MAX_JOBS=${MAX_JOBS:-32} \
  python3 -m pip install -e . --no-build-isolation -q 2>&1 | tail -5
python3 -c 'import lightop; print("import OK")' || {
  note "stage3 FAIL  shipping extension does not import"; echo "$VERDICT"; exit 1; }
python3 - <<'PY'
import torch, lightop, pathlib
ref = pathlib.Path('/tmp/ref_logits.pt')
torch.manual_seed(0)
print('shipping extension ready')
PY
note "stage3a PASS shipping extension built"

echo "--- capturing reference logits from the shipping path ---"
python3 "$REPO/csrc/paged_mqa_logits/diagnostics/fp4_reference_capture.py" --save /tmp/ref_logits.pt || {
  note "stage3b FAIL reference capture failed"; echo "$VERDICT"; exit 1; }
note "stage3b PASS reference logits captured"

echo "--- rebuilding with the native path + strict fallback ---"
LIGHTOP_PAGED_FP4_NATIVE=1 LIGHTOP_PAGED_FP4_NATIVE_STRICT=1 \
  PYTORCH_ROCM_ARCH=gfx946 MAX_JOBS=${MAX_JOBS:-32} \
  python3 -m pip install -e . --no-build-isolation -q 2>&1 | tail -5
python3 -c 'import lightop; print("import OK")' || {
  note "stage3c FAIL native extension does not import"; echo "$VERDICT"; exit 1; }
note "stage3c PASS native extension built (strict: fallback traps)"

# ---------------------------------------------------------------- stage 4
say "4  differential: native vs shipping"
# Strict mode means a fallback aborts the process, so surviving this stage is
# itself proof the native MMAC ran on every tile.
if python3 "$REPO/csrc/paged_mqa_logits/diagnostics/fp4_reference_capture.py" \
     --compare /tmp/ref_logits.pt; then
  note "stage4 PASS native output matches the shipping path, and strict mode"
  note "            proves no tile fell back to the decoded path"
else
  rc=$?
  if [ $rc -ge 128 ]; then
    note "stage4 FAIL process aborted -- strict mode trapped, i.e. some tile was"
    note "            ineligible for the native path (check the scale range gate)"
  else
    note "stage4 FAIL native output differs from the shipping path"
  fi
fi

say "5  benchmark"
python3 test/benchmarks/benchmark_paged_mqa_logits_fp4.py 2>&1 | tail -25 \
  && note "stage5 DONE  compare against the shipping numbers recorded in stage 3"

say "verdict"
echo "$VERDICT"
echo
echo "log saved to $LOG"
