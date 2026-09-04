#!/usr/bin/env bash
set -u
source ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/code
L=${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/logs
while pgrep -f 'run_chain2.sh' >/dev/null 2>&1; do sleep 20; done
echo "=== chain3 start $(date '+%F %T') ==="
python q26_finite_difference.py > "$L/block21_fd.log" 2>&1; echo "fd exit=$? $(date '+%T')"
echo "=== chain3 done $(date '+%F %T') ==="
