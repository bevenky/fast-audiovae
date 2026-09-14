# A4: direct projected-history phase assembly

Experiment only. No production source, projections, trained constants or state
representation changes. The source is a portable adaptation of the maintained
Apple phase operator's scalar loop, without its NEON implementation or Accelerate
dependencies. It uses two ordered FP32 adds and chronological output stores.

`prepare.py` authenticates an embedded-weight FP32 streaming graph using the
required SHA argument. By default it extracts the unique C1024/stride8 region;
`--node` can explicitly select another actual region. It verifies the exact
five-node wrapper, exclusive intermediate consumers, positive-zero padding,
constant bias, slices and external projected-history endpoints. It emits a full
**unqualified** overlay and matched regional graphs; all original initializers,
graph inputs/outputs and projection nodes remain unchanged. Root must bind the
FP32 control's provenance separately: this operator does not inspect or replace
matrix precision elsewhere in the model.

The eliminated operations are two full projection concatenations, one retained
history slice, the cropped leading phase output, and dispatch between those
nodes. The previous projection was already cached. No GEMM, Snake or convolution
work is eliminated. State remains `[1,C*stride,1]` containing the final unshifted
previous projection. T0 copies the incoming state and emits empty audio.

Schema: `fast.audiovae.amd.phase.state.v1::StatefulPhaseFinishF32` takes
`current[1,C*S,T], previous[1,C*S,T], history[1,C*S,1], fixed_bias[C]` and returns
`audio[1,C,T*S], next_history[1,C*S,1]`. Attributes are `phase_state_abi=1`,
`threads=1`, `channels=C`, `stride=S`. The candidate has one native worker;
baseline phase dispatch retains its existing ORT worker policy. A four-thread
session comparison therefore measures the complete scheduling choice as well as
the removed copies. It must not be described as an isolated copy-only speedup.

Build only after root authorizes compilation on the AMD machine, with the same
ORT C/C++ headers as the control. No system install is needed:

```sh
g++ -O3 -std=c++17 -shared -fPIC -fvisibility=hidden -fno-fast-math \
  -ffp-contract=off -pthread -I "$ORT_INCLUDE" native_ops.cpp \
  -o liba4_phase.so
```

Record compiler/version, command, header hashes, source SHA and final library SHA
in the parent build receipt. There is no BLAS dependency or intrinsic ISA
requirement. Do not add `-Ofast` or reassociation flags.

Preparation (no inference):

```sh
python prepare.py --source "$FP32_STREAM_GRAPH" \
  --expected-source-sha256 "$FP32_STREAM_SHA" --output prepared-001
```

Root-controlled single screen (no automatic repeat or promotion):

```sh
python screen.py --plan prepared-001/plan.json \
  --baseline-library "$BASE_PHASE_LIBRARY" --baseline-sha256 "$BASE_PHASE_SHA" \
  --candidate-library ./liba4_phase.so --candidate-sha256 "$A4_LIBRARY_SHA" \
  --ort-version 1.30.0 --threads 1 --time --output screen-001
```

The screen runs all math first, using the real trained bias and finite projection
fixtures. It checks FP32 bitwise audio/history against both the original five-node
region and an independent ordered oracle: zero/quiet/signed inputs, nonzero
history, empty/odd/40/80 ms lengths, cancellation, read-input aliasing, independent
streams, reset, future extension, and partitions with each arm's own state.
It times both packet sizes with two warmups and six balanced randomized pairs,
16 complete calls per arm/pair. Every raw call duration is saved; no outliers are
removed. Selection is per size: median reduction >=10% and six of six wins.
All `session.run` calls, including correctness/warmups, are charged to a 2-second
native-call cap; measured calls also share a 1-second cap. Compilation, session
creation, source hashes and NumPy oracle/comparisons are excluded and are not RTF.
Errors retain a failed result; output directories cannot be overwritten.

After a passing decision, add `liba4_phase.so` to the experimental bundle's
registered custom libraries and select the prepared overlay as its streaming
model. Preserve its original full model and state metadata. Hash every copied
runtime dependency through the parent recipe. The overlay is not yet qualified
for complete-decoder use until the parent runs its numerical/state checks.
