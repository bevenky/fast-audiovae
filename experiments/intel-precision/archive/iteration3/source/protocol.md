# Exact-output Intel decoder comparison

This is a new paired experiment against the frozen `int8_large` decoder.
It does not compare approximate INT8 against FP32 or repeat perceptual scoring.
The candidate must reproduce the accepted INT8 waveform bytes.

Use the existing public `benchmarks/compare_decoders.py` helpers and pin that
file. Supply two configs in its public schema. The baseline config and completed
campaign are the unchanged five-model precision campaign. Copy the baseline
config for the candidate, preserve its complete corpus and `int8_large` model,
then append a differently named candidate. Update relative paths and all hashes.
Both graph external-data inventories must be exact. Include transitive native
core libraries in `artifact_sha256`, even when ORT registers only their bridges.
The candidate's ORT domains and native symbols must coexist with the baseline.

```sh
CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void \
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE \
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python validate_candidate.py \
  --baseline-config baseline.json --baseline-config-sha256 BASELINE_SHA \
  --completed-campaign completed-results.json --completed-campaign-sha256 CAMPAIGN_SHA \
  --candidate-config candidate.json --candidate-config-sha256 CANDIDATE_SHA \
  --candidate-model iteration3 \
  --harness compare_decoders.py --harness-sha256 HARNESS_SHA \
  --script-sha256 SCRIPT_SHA --output-dir fresh-full-results
```

The process requires Linux x86, ORT1.29.0, two ORT workers and logical CPU
affinity0,1. oneMKL and OpenMP remain single threaded. No GPU libraries may be
mapped. Visibility variables must be set before process start. Session loading,
warmup, validation, logging and output hashing are outside inference timers.

Full mode reexecutes both decoders on all60 original clips. Baseline outputs
must match the completed campaign's untrimmed mono48k FLOAT WAVs, whose file and
sample hashes are verified. Candidate outputs must match those same samples
through uint32 views, including signed zero. Latent bytes must be unchanged.
No waveform is normalized, cropped or reencoded. Since outputs must be exact,
this experiment writes hashes rather than duplicating hundreds of audio files.

Fixed additional checks use a257-frame prefix from the longest frozen clip:
short lengths1/2/3/7/8/15/16/17/31/32/33/63/64/65/127/128/129/255/256,
future perturbations at1/7/8/16/31/32/63/64/127/128/255/256, repeat and
long-short-long checks, and concurrent257/33-frame calls on each same session.
The original short-versus-long prefix gate remains atol1e-5/rtol1e-4 for both
models. It was never a bitwise length-invariance guarantee. Every candidate
short output additionally must match a fresh baseline output at that exact
input length bitwise. Repeat, same-length future invariance, concurrency replay
and complete waveforms retain uint32 equality. A failing baseline probe stops
the experiment; no tolerance is changed after results are seen.

Timing retains the original ten UIDs and full padded generated duration.
Each model/clip receives two fresh warmups and five fresh timed calls. Adjacent
A/B pairs have deterministic randomized order, balanced two-versus-three over
five repetitions per clip. Clip order is randomized per repetition. Every timed
output is checked against its validated reference after its timer stops. Full
mode therefore contains100 timed calls and40 warmup calls. No cached waveform
stands in for timed inference.

Pass `--screen` for a separate, non-promoting experiment. It uses the original
Hindi, English and Brazilian Portuguese timing UIDs, validates those three
complete waveforms, and uses a65-frame boundary probe with all listed short
lengths below65 and perturbations at8/32/64. It still uses two warmups and five
paired repetitions, producing30 timed calls and12 warmups. A screen can reject
a candidate early; it cannot establish a full-cohort result or set accepted.

Promotion requires complete full-mode validation, at least10% lower aggregate
elapsed time, and no slower mean among the ten timed clips. Per-repeat aggregate
reductions, per-clip dispersion and a paired clip-bootstrap interval are
reported separately. No clip or repeat is removed after seeing results.
Host steal/load, cgroup snapshots and process resource use contextualize noise;
they do not automatically excuse a failed gate. Recheck the full fixed cohort
if another workload disturbed the measurement.

Every run needs a fresh output directory. Resume is intentionally unsupported.
Preflight failures stop before creating output. Later failures produce
`status: failed`, keep completed check/timing records and cannot be accepted.
Config, model, sidecar, library, corpus, harness, script,
campaign and reference-file hashes are rechecked before sessions, before timing
and after timing. The baseline campaign is inspected for complete validation,
timing and CPU-provider evidence. This script has not itself executed any
candidate until a resulting report records those fresh calls.
