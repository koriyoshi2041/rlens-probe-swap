#!/usr/bin/env bash
set -u
source ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/code
L=${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/logs
while pgrep -f 'run_chain5.sh' >/dev/null 2>&1; do sleep 20; done
echo "=== chain6 start $(date '+%F %T') ==="
python q28_direction_control.py > "$L/block24_dircontrol.log" 2>&1; echo "dircontrol exit=$? $(date '+%T')"
echo "=== chain6 done $(date '+%F %T') ==="
