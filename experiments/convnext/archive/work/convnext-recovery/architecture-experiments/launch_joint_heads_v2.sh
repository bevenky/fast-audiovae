#!/usr/bin/env bash
set -euo pipefail

# Run on the existing H100 pod, after staging the v2 harness and its tests.
experiment_root=/tmp/fast-audiovae-recovery-20260909
remediation_root=/workspace/fast-audiovae-convnext-20260909-r9/remediation
export PYTHONPATH="$experiment_root/joint-head-code-v2:$experiment_root/code214:$experiment_root/code:$experiment_root/head-experiments-code-v2:$experiment_root/peak-silence-code-v1:$experiment_root/optimizer-causal-code-v1:$remediation_root/corrected-code:$remediation_root/diagnostic-code:$remediation_root/diagnostic-supplement-code"

"$experiment_root/venv214/bin/python" -m pytest -q "$experiment_root/joint-head-code-v2/test_run_joint_heads.py"
exec "$experiment_root/venv214/bin/python" -u "$experiment_root/joint-head-code-v2/run_joint_heads.py" \
  --checkpoint "$experiment_root/corrected-update-400-v1/quarter_rate/final.pt" \
  --training-receipt /dev/shm/fast-audiovae-recovery-fresh12800-v1/receipt.json \
  --canonical-receipt "$experiment_root/canonical-panel-v1/receipt.json" \
  --canonical-inventory "$experiment_root/canonical-panel-v1/source-inventory.json" \
  --out "$experiment_root/joint-head-silence-v2"
