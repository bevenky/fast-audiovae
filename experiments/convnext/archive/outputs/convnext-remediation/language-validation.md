# Expanded language validation at step 3000

Completed on the Runpod CPU with one thread. This adds 52 reserved recordings across 13 language groups, including the nine previously missing Indic languages plus Mandarin, Cantonese, Arabic and German. Together with the existing panel, there are now validation recordings for all 22 scheduled Indic languages. This is a separate immutable appendix, so the original 167-crop trend remains comparable over time.

The 485.213 seconds of source audio produced 102 fixed startup/tail crops and 160.621 scored seconds. These are selected windows, not an exhaustive evaluation of every source sample. Four recordings per language give a useful initial check, not broad population coverage.

| Aggregate at step 3000 | Result |
|---|---:|
| Active waveform correlation against teacher | 0.9115 |
| Normalized waveform error | 0.2132 |
| Mel error | 1.3497 |
| Active crops reaching 0.99 correlation | 6/93 |
| Quiet windows passing current checks | 0/1629 |
| Samples at or beyond full scale | 16 |
| Maximum student amplitude | 1.1370 |

The nine new Indic groups have active correlation **0.9215** in aggregate; the four FLEURS cohorts have **0.8829**. Quiet reconstruction and peak control remain unresolved. The 16 samples at or beyond full scale occur in one Konkani startup crop. The output was measured without clipping, gain normalization or fitted alignment. These are waveform and spectral diagnostics, not perceptual scores.

| Language | Active / total crops | Correlation ↑ | Waveform error ↓ | Mel error ↓ |
|---|---:|---:|---:|---:|
| Bodo | 8/8 | 0.915 | 0.205 | 1.958 |
| Dogri | 8/8 | 0.955 | 0.174 | 1.229 |
| Konkani | 8/8 | 0.909 | 0.212 | 1.947 |
| Kashmiri | 7/7 | 0.923 | 0.194 | 1.359 |
| Maithili | 8/8 | 0.952 | 0.182 | 1.201 |
| Manipuri | 7/7 | 0.937 | 0.203 | 1.251 |
| Odia | 8/8 | 0.827 | 0.296 | 1.693 |
| Sanskrit | 8/8 | 0.933 | 0.214 | 1.306 |
| Santali | 7/8 | 0.948 | 0.175 | 1.971 |
| Mandarin | 3/8 | 0.795 | 0.326 | 1.040 |
| Arabic | 8/8 | 0.823 | 0.256 | 0.903 |
| German | 8/8 | 0.970 | 0.130 | 0.844 |
| Cantonese | 5/8 | 0.892 | 0.201 | 0.834 |

Mandarin has only three active crops and Cantonese five because the fixed startup/tail selection includes substantial quiet audio. Do not treat the table as a language-quality ranking. Future coverage can add a separate interior-speech panel while preserving this appendix's selected windows and baseline.

## What was verified

All source-file hashes and decoded mono 16 kHz lengths matched the immutable manifest. The source identity inventory was checked independently against all 320,000 optimization and 512 calibration windows, including future training. The acquisition audit additionally checked 15 full exclusion snapshots and confirmed unique source hashes, receipts and official held-out partitions.

The 36 IndicVoices recordings use the official valid partition, with 36 known speakers and sessions. The 16 FLEURS recordings use the official validation partition. FLEURS does not provide verified speaker/session identities here; its unknown-person placeholders are explicitly excluded from claims of speaker disjointness. Exact hashes and supplied canonical identities do not detect unknown re-encodings or acoustic duplicates. Language labels are official source metadata and were not independently confirmed by listening.

The pinned original AudioVAE2 source and checkpoint generated continuous complete-utterance CPU FP32 targets. The source and weights were verified, and the teacher's state hash matched the training teacher. Raw 64-channel posterior means, original gain, true causal history and exact valid sample lengths were preserved. Teacher and student states were unchanged after evaluation. **Zero optimizer updates occurred**, and the original target-file SHA-256 was unchanged. The live training process and its dataset plan were not modified.

The teacher cache was capped at 128 MiB with a 4 GiB free-space reserve. Only JSON reports were copied locally. Audio, targets and checkpoints remain on Runpod. CPU targets have their own provenance; their identity is not assumed to match the original panel's CUDA target-preparation backend.

## Evidence

- [Compact evaluation results](language-validation-summary.json)
- [Immutable panel identity](language-validation-identity.json)
- [Independent acquisition and exclusion checks](language-validation-acquisition-checks.json)
- [Frozen teacher target identity](language-validation-target-identity.json)

Panel identity: `43bbd4a03fafa51e8ad783c969f0a5d67ae37ab3a0af0bc11d6f87a0807329f1`.

Student checkpoint: `027387045c28a60883d2696568156ad1dbaa971dbb4e6086be765a486363c8a2`.

Original target-file SHA-256, verified unchanged: `3349f715c9ae4c4b7ac859cde935cdc949395bfc0e15915ad759c8e547a4dd2e`.

The full evaluation is saved on Runpod at `/workspace/fast-audiovae-convnext-20260909-r9/remediation/language-validation/evaluation-step003000.json`. Publish this under its own language-panel label, not inside the old aggregate time series. These new validation files must also be included in the exclusions for any future training-data handoff.
