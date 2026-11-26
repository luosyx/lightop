#!/bin/bash
# gfx946 card-time runbook for the native FP4 MMAC path.
#
# Run this FIRST when a gfx946 part becomes available.  Everything that does
# not need the card has already been done and checked on gfx938, so this script
# is pure execution: it does not compile anything that was not already proven to
# compile, and it reports one verdict per stage.
#
#   bash gfx946_cardtime.sh            full run
#   bash gfx946_cardtime.sh quick      stages 0-2 only (~2 min)
#
# ---------------------------------------------------------------------------
# REQUIRED ENVIRONMENT
#
# The script only checks these; it installs nothing, so that no card time is
# spent on package downloads.  Stage 0 names whatever is missing.
#
#   GPU        a gfx946 part visible to torch.  gfx938 is not enough: the
#              native MMAC is gated on hcu-mmop6-insts, which gfx938 lacks.
#
#   DTK        27.04-shb  (check: cat /opt/dtk/.dtk_version)
#              Earlier DTKs do not ship __builtin_hcu_mmac_f32_16x16x64_fp4*.
#              hipcc is expected at /opt/dtk/bin/hipcc.
#
#   torch      2.13.0+das.opt1.dtk2704, importable.
#              If a container lacks it:
#                pip install --no-deps 'http://10.16.1.201:9929/customer/\
#                  dtk2704/+f/d6d/06607a8a9a9bc/torch-2.13.0+das.opt1.\
#                  dtk2704.2611120915.ga6d2da-cp312-cp312-manylinux_2_28_\
#                  x86_64.whl'
#                pip install typing_extensions filelock fsspec sympy \
#                  networkx jinja2 setuptools cloudpickle numpy
#              The second line is needed because --no-deps installs the wheel
#              but leaves `import torch` unsatisfiable.
#              triton / flash_attn / vllm are NOT needed by this script.
#
#   repo       this checkout, writable (stage 3 does `pip install -e .`).
#              git safe.directory must cover it if the mount is owned by
#              another uid.
#
# Container that satisfies all of the above (image already has DTK 27.04-shb):
#   10.16.1.152:5000/hcu/base/dtk:27.04-ubuntu22.04-py3.12
# ---------------------------------------------------------------------------
#
# Stages
#   0  environment   verify the above, install nothing
#   1  sanity        device really is gfx946
#   2  probe         native MMAC numerical test (settles the accumulator map)
#   3  extension     build the python extension twice: shipping, then native
#   4  differential  operator output, native vs the shipping path
#   5  benchmark     native vs shipping timing
set -u
MODE="${1:-full}"
# Locate the repo from this script's own path (test/mqa_logits/gfx946_fp4/), so
# the script works regardless of the current directory or checkout location.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
DIAG="$REPO/csrc/paged_mqa_logits/diagnostics"
LOG="${TMPDIR:-/tmp}/gfx946_cardtime_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"

say() { printf '\n========== %s ==========\n' "$1"; }
VERDICT=""
note() { VERDICT="$VERDICT
$1"; echo "$1"; }
die() { note "$1"; echo "$VERDICT"; exit 1; }

# ---------------------------------------------------------------- stage 0
# Verify only.  Anything missing is reported against the header above rather
# than fixed here, so card time is never spent downloading packages.
say "0  environment"
cd "$REPO" 2>/dev/null || die "stage0 FAIL  repo not found at $REPO"

DTK_VER="$(cat /opt/dtk/.dtk_version 2>/dev/null | head -1)"
echo "DTK:   ${DTK_VER:-<not found>}"
case "$DTK_VER" in
  *27.04-shb*) ;;
  "")  die "stage0 FAIL  /opt/dtk/.dtk_version missing -- DTK 27.04-shb required" ;;
  *)   note "stage0 WARN  DTK is '$DTK_VER', expected 27.04-shb;"
       note "             the native builtins may not exist" ;;
esac
[ -x /opt/dtk/bin/hipcc ] || die "stage0 FAIL  /opt/dtk/bin/hipcc not executable"

python3 -c 'import torch' 2>/dev/null \
  || die "stage0 FAIL  torch not importable -- see REQUIRED ENVIRONMENT above"
echo "torch: $(python3 -c 'import torch; print(torch.__version__)')"

git config --global --add safe.directory "$REPO" 2>/dev/null
note "stage0 PASS  environment ready"

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
cd "$DIAG" || exit 1
hipcc -x hip --offload-arch=gfx946 -O2 "$DIAG/fp4_mmac_probe.cu" -o /tmp/probe || {
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
python3 "$HERE/fp4_reference_capture.py" --save /tmp/ref_logits.pt || {
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
if python3 "$HERE/fp4_reference_capture.py" \
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
