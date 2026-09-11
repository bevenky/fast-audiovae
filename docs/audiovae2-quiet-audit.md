# Quiet-audio audit of the compressed AudioVAE2 decoder

Measured on 2026-09-10. The selected candidate uses the existing five-scale mel objective and AdamW at 3e-5. The bounded pilot is complete at 1,000 updates, using 3,000 distinct sources and 2.052 hours of scored audio. Its checkpoint and optimizer state are retained on Runpod. No architecture or loss change was made during this audit.

## What changed despite retaining the architecture

The student keeps the teacher's layer types, all nine residual units in stages 2–4, Snake, causal geometry, conditioning, final waveform convolution and tanh. It narrows two internal boundaries from 512 to 256 channels and from 256 to 128. Removing channels changes the matrices and the function they compute. Keeping the original input and output shapes does not preserve the pretrained mapping.

Both paths still receive identical 64-channel encoder latents and identical stage-1 output. On the fixed 12-source boundary panel, prefix error is exactly zero. The first fully comparable compressed boundary, the complete stage-4 output, has quiet-region NRMSE 0.582 at step 256 and 0.408 at step 1,000. The frozen suffix consequently receives an imperfect representation. Stage-2/3 selected-coordinate traces are diagnostic proxies, not full-layer equivalence tests.

Tanh is effectively linear at these quiet amplitudes. The difference between pre-tanh and final quiet residual RMS is less than 4e-10. It bounds loud output but cannot remove this low-level mismatch.

## Same-source progress

These results use the same 96 development sources, including all 22 scheduled Indic languages and expressive material. Active correlation averages 94 sources with teacher-active windows; waveform and quiet errors pool valid samples. The near-silence subset contains 184 windows with teacher RMS at most 1e-5.

| Measurement | Step 256 | Step 1,000 |
|---|---:|---:|
| Active waveform cosine | 0.91328 | 0.94468 |
| Waveform MAE | 0.008732 | 0.006786 |
| Common five-scale mel | 0.65827 | 0.50550 |
| Full stage-4 MSE | 0.041361 | 0.024986 |
| Quiet residual RMS | 0.0003025 | 0.0002442 |
| Quiet-window failures | 2,542 / 2,544 | 2,429 / 2,544 |
| Near-silence residual RMS | 0.00006275 | 0.00002805 |
| Near-silence failures | 184 / 184 | 93 / 184 |

Waveform MAE improves on 95 of 96 sources, active correlation on all 94, and mel and full-group MSE on all 96. Quiet residual improves in 2,483 of 2,544 windows. This supports continued recovery with the current model; it does not establish eventual parity.

## Three different quiet problems

**Target or cache disagreement was not found.** The live original teacher reproduces all 2,544 cached quiet windows exactly, with zero residual. Recomputed spectral metrics also match the saved reports exactly. The input masks, valid sample counts and source identities agree.

**A large startup mismatch is a teacher artifact.** Across several source-start crops, the largest residual occurs at sample 1,414, or 29.458 ms. In the retained checkpoint the teacher is approximately -0.0085883 and the student +0.00002487. This is mainly a teacher transient that the student does not reproduce. Direct reads of the authenticated Tajik, Luxembourgish and Somali source files confirm that every sample in their first 190 ms is zero. It would be misleading to call this a student-generated spike. Keep teacher reconstruction error visible, but also judge this behavior against the original source silence.

**Genuine added near-silence energy remains.** In the 184 near-silence windows at step 1,000, teacher RMS is 9.56e-6 and student RMS is 2.82e-5. Ninety-three fail. Later quiet error also persists: 1,701 of 1,762 windows after 800 ms or in continuation crops fail. Startup treatment alone cannot fix those windows.

## What the failure count means

The existing engineering checks classify a window as quiet when teacher RMS is at most 1e-3. They require both residual RMS at most `max(0.141421 * teacher_rms, 1e-5)` and output RMS at most `max(1.122018 * teacher_rms, 1e-5)`. These are strict reconstruction and amplitude requirements, not validated audibility thresholds. Quiet windows include low-level nonzero sounds as well as near-silence.

At step 1,000, 1,444 windows fail residual fidelity only, 60 fail amplitude only, and 925 fail both; 115 pass. All 60 amplitude-only windows previously failed both criteria, so they are not newly failing windows. Short partial tails account for only eight windows at step 256 and cannot explain the overall result.

Per-window residual means explain 1.64% of total quiet residual energy and 7.63% in near-silence at step 1,000. Those shares were 4.51% and 56.76% at step 256. A universal offset subtraction would not address most remaining error. This decomposition does not by itself distinguish varying noise from low-level phase or gain mismatch.

## Data and loss audit

Quiet audio was not omitted: 15.14% of the first 768 fitting sources' scored samples are quiet by the same teacher-conditioned definition. Near-silence at RMS at most 1e-5 contributes 20.92 seconds in those sources and another 45.79 seconds in the next 2,232. Its proportion is only 1.11% and 0.83%, respectively. Presence is verified; adequacy across every quiet state is not established. No exact-zero decoded teacher windows were observed, which does not imply the source recordings lack digital silence.

Waveform loss is raw valid-sample-pooled L1. Every nonzero residual has the same absolute waveform derivative before model backpropagation, irrespective of loudness. The earlier inverse-RMS example amplification is absent. Whole-group MSE includes quiet feature cells, with correct weights for partial cells. Log-mel uses a 1e-5 floor; floor occupancy and late quiet-versus-active parameter-gradient balance have not been measured. Neither is established as the cause of the remaining error.

## Retained decision and remaining guard

Retain the current-loss, 3e-5 checkpoint. It also beats the reference-loss lower-rate arm on the added reference seven-scale metric, 1.47469 versus 1.48510 at the matched 256-update endpoint. This is the best supported continuation from a short, single-seed comparison, not a proven optimal recipe.

Continue the same recipe on a fresh source-unique block before changing architecture or introducing a silence gate. Keep source-referenced startup behavior, near-silence output, later quiet residual and expressive gain separate. One whistling source regresses: its waveform MAE increases 6.96% despite improved correlation, and its active output RMS is 34.5% above the teacher. Correlation and aggregate progress do not waive that failure.

The final continuation report archive SHA-256 is `ac8b8334828d268423f1b595b27806f315a7620d3b232e471d0884c652e13a3f`; the retained checkpoint SHA-256 is `bfecd759eb58db1eeeca3dd693660b26be91bd69a93e3457b9d91523c5de0573`. Per-window evidence, source-prefix checks, data histograms and boundary reports are retained alongside the experiment outputs. Training is stopped at the planned 1,000-update limit pending the next continuation decision.
