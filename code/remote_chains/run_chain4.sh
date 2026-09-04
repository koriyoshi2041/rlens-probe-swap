#!/usr/bin/env bash
set -u
source ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/activate.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/code
L=${MATS_ROOT:-/root/autodl-tmp/mats-r-lens}/work/logs
while pgrep -f 'run_chain3.sh' >/dev/null 2>&1; do sleep 20; done
echo "=== chain4 start $(date '+%F %T') ==="
python -m pytest tests/test_clamp.py -q > "$L/block22_metric_test.log" 2>&1; echo "metric test exit=$? $(date '+%T')"
python q23_rederivation.py > "$L/block18b_rederivation.log" 2>&1; echo "rederivation-fixed exit=$? $(date '+%T')"
python q25_specificity.py  > "$L/block20b_specificity.log" 2>&1; echo "specificity-fixed exit=$? $(date '+%T')"
echo "=== chain4 done $(date '+%F %T') ==="
