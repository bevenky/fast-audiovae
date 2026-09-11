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
| Hash-verified frozen AudioVAE2 loader | H100 target parity and sustained training |
| Teacher spectral/waveform warmup losses | Corpus sampling, GAN trainer and perceptual training |
| Resumable AdamW/Muon preflight and TensorBoard logging | Muon versus AdamW speech convergence comparison |
| Commercial-source quotas and manifest leakage checks | Downloaded training corpus and acoustic fingerprinting |

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

Validate the proposed data mixture:

```bash
.venv-convnext/bin/python -m audiovae_student.data validate-mixture experiments/convnext/configs/data-mixture.json
```

Training manifests must pass source, bandwidth, split and identity checks.
Checks do not discover unknown speaker identity or acoustically duplicate
encodings. Keep source audio, model assets and checkpoints outside Git.

See [the training plan](../../docs/convnext-training-plan.md) and
[optimizer decision](../../docs/convnext-optimizer.md) for the staged work.
