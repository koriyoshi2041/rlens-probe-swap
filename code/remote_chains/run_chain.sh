#!/usr/bin/env bash
set -u
source ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/code
L=${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/logs
while pgrep -f q22_fit_ladder >/dev/null 2>&1; do sleep 20; done
echo "=== chain start $(date '+%F %T') ==="
python q24_eval_ladder.py   > "$L/block19_ladder_eval.log" 2>&1; echo "eval_ladder exit=$? $(date '+%T')"
python q21_damping.py       > "$L/block16_damping.log"     2>&1; echo "damping exit=$? $(date '+%T')"
python q23_rederivation.py  > "$L/block18_rederivation.log" 2>&1; echo "rederivation exit=$? $(date '+%T')"
echo "=== chain done $(date '+%F %T') ==="
