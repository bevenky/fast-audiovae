# Lightweight AudioVAE2 decoder

An experimental causal decoder trained from fresh weights with the original
AudioVAE2 as teacher. It accepts the same unscaled 64-channel, 25 Hz latents and
emits 48 kHz audio. The existing encoder and released runtime stay unchanged.

The model has ten causal ConvNeXt blocks at 100 Hz, 512 hidden channels,
2048-channel expansions and a direct waveform head. No Supertonic code,
weights, normalization statistics or generated targets are used.

## Current implementation

| Available | Not yet qualified |
|---|---|
| Fresh trainable decoder and explicit streaming state | Trained speech quality or speed relative to Mimi |
| Full and streaming ONNX export, CPU runner | Production runtime integration |
| Frozen AudioVAE2 loader and H100 target parity | Final trained speech quality |
| Cached-speech warmup with held-out reconstruction checks | GAN trainer and perceptual training |
| Resumable AdamW/Muon training and TensorBoard logging | Matched optimizer convergence comparison |
| Pinned speech acquisition and exact-identity leakage checks | Completed 500-hour corpus and acoustic fingerprinting |

The CLI preflight uses synthetic audio and a reduced-width test model. It tests
training and checkpoint wiring, not audio quality. The default `StudentDecoder`
and untrained export use the full architecture.

## Run locally

From the repository root, use a separate environment:

```bash
uv venv --python 3.12 .venv-convnext
uv pip install --python .venv-convnext/bin/python -r experiments/convnext/requirements.txt
export PYTHONPATH=experiments/convnext
.venv-convnext/bin/python -m pytest experiments/convnext/tests -q
.venv-convnext/bin/python -m audiovae_student.training --steps 12 --checkpoint experiments/convnext/checkpoints/preflight.pt
```

The requirements are separate from the normal installation. Run equivalent
`python -m venv` and `python -m pip install -r ...` commands if using pip.
ONNX Runtime must be 1.29.0 for the experimental CPU runner. Exact transitive
versions from the validated environment are recorded in `requirements.lock`.

Export and check the complete untrained architecture:

```bash
.venv-convnext/bin/python -m audiovae_student.export --untrained --output experiments/convnext/artifacts/student --seed 17
.venv-convnext/bin/python -m audiovae_student.validation --bundle experiments/convnext/artifacts/student --output experiments/convnext/artifacts/validation.json
```

Exports require a new destination. They include graph checksums, initialization
provenance and explicit histories. Empty chunks and reset/flush are handled by
the streaming wrapper. Each latent frame emits 1920 samples; the low-level
decoder cannot infer a partial original input length. The teacher wrapper's
`reconstruct` method trims to exactly three times the original 16 kHz length.

Set `AUDIOVAE2_SOURCE` and `AUDIOVAE2_CHECKPOINT` to trusted local assets to enable
the actual-teacher integration test. Both must match the pinned hashes in
`audiovae_student/teacher.py`. The teacher loader never downloads models.

## Monitor training

Install `tensorboard==2.21.0` in the training environment, then enable logging:

```bash
.venv-convnext/bin/python -m audiovae_student.training --steps 32 --checkpoint experiments/convnext/checkpoints/preflight.pt --log-dir experiments/convnext/runs --run-name preflight-adamw
.venv-convnext/bin/tensorboard --logdir experiments/convnext/runs
```

Open the URL printed by TensorBoard. The Scalars tab shows real loss components,
learning rates, gradient norm and step time. Select multiple runs to compare
their curves. Resume with the same run name and `--resume` checkpoint; stale
events after that checkpoint are removed from the displayed history.

The current CLI is a synthetic setup test. It produces no validation or speech
quality scores. Muon requires a PyTorch build with native `torch.optim.Muon`;
use `--optimizer muon_adamw` with a separate run name and checkpoint.

## First speech warmup

The initial shard contains 230 training utterances from LibriSpeech and FLEURS
across 12 languages, plus 25 speaker-disjoint LibriSpeech development utterances.
This is a completed 45.85-minute training bootstrap, not the expanded corpus.
All 80 earlier evaluation clips are excluded by source identity and split.

`audiovae_student.acquire` downloads official source splits.
`audiovae_student.prepare_targets` verifies the audio and caches whole-utterance
FP32 latents and waveforms from the frozen original encoder and decoder.
It warms up the original TorchScript path and records the execution policy.
No teacher weights or gradients are updated.

With the verified cache index prepared, start the full student on an explicit
CUDA training environment:

```bash
python -m audiovae_student.corpus_training --index target-cache/index.json --output-dir training-runs/warmup --device cuda --steps 1000 --optimizer muon_adamw --log-dir runs --run-name warmup-speech
```

The trainer updates only the new decoder. It samples languages, accumulates
eight examples per update, excludes context and padding from losses, and saves
checkpoints every 100 updates. Resume with the same arguments and
`--resume training-runs/warmup/latest.pt`. TensorBoard exposes `train/total` and
`validation/total`. Development loss uses fixed first-2.56-second crops; it is
not a multilingual perceptual-quality score. Inference benchmarks remain CPU-only.

## Expanded speech run

The next phase continues the saved step-1,000 student and optimizer to step
10,000 on a fixed corpus of at least 500 fresh training hours. Acquisition is
in progress. It includes all 102 FLEURS configurations, LibriSpeech, original
IndicVoices across 22 scheduled Indian languages, and reviewed expressive data.
Previously trained utterances and held-out clips are excluded.

Speech acquisition supports concurrent source archives with `--workers 4`.
Workers share the download and storage limits; completed source manifests are
preserved when resuming. IndicVoices has its own bounded worker pool. Its
`--download-method native` option stages one pinned Parquet file per worker,
verifies its checksum and releases temporary storage after reading it. Audio
and teacher caches stay on the training machine.

`audiovae_student.source_training` uses true batches of 64 and scores each
segment once. Short expressive clips retain their actual length. Causal context
can overlap, but it is excluded from scored exposure and loss. The sampler stops
if fresh segments run out. It never silently starts another epoch.

The original encoder and teacher decoder stay frozen in FP32. A bounded cache
generates their targets from whole utterances. This avoids storing hundreds of
gigabytes of teacher audio. Only the student receives optimizer updates.
TensorBoard separates data preparation time, student time, loss and unique
audio exposure. Reconstruction validation runs at the end of this phase;
perceptual quality and CPU latency still require separate qualification.

The larger batch measured 284 examples/second against 58 for the earlier serial
accumulation on the existing H100. This measures the training step with prepared
targets, not download, teacher preparation or complete run throughput.

Validate the proposed data mixture:

```bash
.venv-convnext/bin/python -m audiovae_student.data validate-mixture experiments/convnext/configs/data-mixture.json
```

Training manifests must pass source, bandwidth, split and identity checks.
Checks do not discover unknown speaker identity or acoustically duplicate
encodings. Keep source audio, model assets and checkpoints outside Git.

See [the training plan](../../docs/convnext-training-plan.md) and
[optimizer decision](../../docs/convnext-optimizer.md) for the staged work.
