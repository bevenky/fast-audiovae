#!/usr/bin/env bash
set -euo pipefail
cd /var/tmp/fast-audiovae-intel-precision/iteration3
export CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE
export PYTHONPATH=/var/tmp/fast-audiovae-intel-precision/profile-support:/var/tmp/fast-audiovae-fusion-20260907/vendor:/var/tmp/fast-audiovae-intel-iteration2/repo/src:/var/tmp/fast-audiovae-intel-iteration2/repo/benchmarks:/var/tmp/fast-audiovae-intel-precision/audio-vendor
unset LD_PRELOAD MKL_CBWR MKL_ENABLE_INSTRUCTIONS LIBXSMM_TARGET
CPU_PYTHON=/var/tmp/fast-audiovae-20260907/venv/bin/python
ITERATION_CONFIG=$1
ITERATION_NAME=$2
ITERATION_OUTPUT=$3
shift 3
ITERATION_CONFIG_SHA=$("$CPU_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$ITERATION_CONFIG")
ITERATION_SCRIPT_SHA=$("$CPU_PYTHON" -c 'import hashlib; print(hashlib.sha256(open("validate_candidate.py","rb").read()).hexdigest())')
"$CPU_PYTHON" validate_candidate.py \
  --baseline-config /var/tmp/fast-audiovae-intel-precision/campaign-int8-r1.json \
  --baseline-config-sha256 d1911542efe3a36261799d90da5c787adb6674f95e126a97648b079b41d1fc89 \
  --completed-campaign /dev/shm/fast-audiovae-intel-precision/campaign-int8-r1/results.json \
  --completed-campaign-sha256 737d8d33954961e32e00287ac71ea9158176c70ed13c76763b59529aadaf5228 \
  --candidate-config "$ITERATION_CONFIG" --candidate-config-sha256 "$ITERATION_CONFIG_SHA" \
  --candidate-model "$ITERATION_NAME" \
  --harness /var/tmp/fast-audiovae-intel-iteration2/repo/benchmarks/compare_decoders.py \
  --harness-sha256 593dc1df8b7ac7334add9211da869eac931aaf61d360656b15a3196467cecf01 \
  --script-sha256 "$ITERATION_SCRIPT_SHA" --output-dir "$ITERATION_OUTPUT" "$@"
