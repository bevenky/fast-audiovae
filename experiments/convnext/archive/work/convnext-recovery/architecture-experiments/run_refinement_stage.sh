#!/usr/bin/env bash
set -euo pipefail

experiment_root=/tmp/fast-audiovae-recovery-20260909
remediation_root=/workspace/fast-audiovae-convnext-20260909-r9/remediation
retained_root=/workspace/fast-audiovae-convnext-20260909-r9/retained-candidates/joint-spectral-step8890-256-20260910
code_root="$experiment_root/refinement-code-v2"
case "$1" in
  teacher) experiment_script=run_teacher_refinements_v2.py ;;
  late) experiment_script=run_late_refinements.py ;;
  *) exit 2 ;;
esac
shift
export PYTHONPATH="$code_root:$experiment_root/spectral-head-code-v1:$experiment_root/joint-head-code-v2:$experiment_root/code214:$experiment_root/code:$experiment_root/head-experiments-code-v2:$experiment_root/peak-silence-code-v1:$experiment_root/optimizer-causal-code-v1:$remediation_root/corrected-code:$remediation_root/diagnostic-code:$remediation_root/diagnostic-supplement-code"
exec "$experiment_root/venv214/bin/python" -u "$code_root/$experiment_script" \
  --checkpoint "$retained_root/parent-step8890.pt" \
  --candidate-heads "$retained_root/joint_spectral-heads.pt" \
  --training-receipt /dev/shm/fast-audiovae-recovery-fresh12800-v1/receipt.json \
  --canonical-receipt "$experiment_root/canonical-panel-v1/receipt.json" \
  --canonical-inventory "$experiment_root/canonical-panel-v1/source-inventory.json" \
  "$@"
