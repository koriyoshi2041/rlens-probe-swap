#!/usr/bin/env bash
set -u
source ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/code
L=${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/logs
while pgrep -f run_chain.sh >/dev/null 2>&1; do sleep 20; done
echo "=== chain2 start $(date '+%F %T') ==="
python q25_specificity.py > "$L/block20_specificity.log" 2>&1; echo "specificity exit=$? $(date '+%T')"
echo "=== chain2 done $(date '+%F %T') ==="
