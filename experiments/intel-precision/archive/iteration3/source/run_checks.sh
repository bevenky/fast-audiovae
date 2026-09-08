#!/usr/bin/env bash
set -euo pipefail
cd /var/tmp/fast-audiovae-intel-precision/iteration3
export CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE
unset LD_PRELOAD MKL_CBWR MKL_ENABLE_INSTRUCTIONS LIBXSMM_TARGET
CPU_PYTHON=/var/tmp/fast-audiovae-20260907/venv/bin/python
ITERATION_BUILD=${1:-/dev/shm/fast-audiovae-intel-iteration3/build-r3}
ITERATION_OUTPUT=/dev/shm/fast-audiovae-intel-iteration3
NATIVE_CPU=/var/tmp/fast-audiovae-fusion-20260907/repo/.build/x86/libfast_audiovae_x86_7312a0b7f908.so
SLEEF_CPU=/var/tmp/fast-audiovae-20260907/repo/.deps/sleef/lib/libsleef.a
test ! -e "$ITERATION_BUILD/check_rows"
g++ -O3 -std=c++17 -fno-fast-math -ffp-contract=off -Wall -Wextra -Werror \
  -I/var/tmp/fast-audiovae-intel-iteration2/repo/native/x86 \
  pipeline/check_rows.cpp "$NATIVE_CPU" "$SLEEF_CPU" \
  -Wl,-Bsymbolic-functions -Wl,--exclude-libs,ALL -Wl,-z,defs \
  -Wl,-rpath,/var/tmp/fast-audiovae-fusion-20260907/repo/.build/x86 \
  -pthread -lm -o "$ITERATION_BUILD/check_rows"
"$ITERATION_BUILD/check_rows" > "$ITERATION_OUTPUT/row-checks-r1.json"
"$CPU_PYTHON" core/check_workspace.py --library "$ITERATION_BUILD/libintel_precision3_core.so" \
  --reference-library /var/tmp/fast-audiovae-intel-precision/native-large-build-r1/libintel_precision_core.so \
  --symbol-prefix ip3_ --backend 1 --output "$ITERATION_OUTPUT/workspace-checks-r1.json"
