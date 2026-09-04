#!/usr/bin/env bash
set -u
source ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/code
L=${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/logs
while pgrep -f 'run_chain4.sh' >/dev/null 2>&1; do sleep 20; done
echo "=== chain5 start $(date '+%F %T') ==="
python q27_semantic.py > "$L/block23_semantic.log" 2>&1; echo "semantic exit=$? $(date '+%T')"
echo "=== chain5 done $(date '+%F %T') ==="
