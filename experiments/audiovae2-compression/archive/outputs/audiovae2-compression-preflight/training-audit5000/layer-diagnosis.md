# Targeted teacher and student layer diagnosis

The bounded forward-only diagnosis completed on the H100 in 21.0 seconds. It measured six development sources at checkpoints 1,000, 4,500 and 5,000, plus six fresh training sources for target consistency. No optimizer was created, no parameter update occurred, and checkpoint files, teacher tensors and frozen student tensors passed preservation checks. Exact measurements are in [regressing-layer-diagnosis-v1.json](regressing-layer-diagnosis-v1.json).

## Controls and original implementation

All six development crops and all six fresh crops reproduced the cached full-source teacher waveform **bitwise exactly on every valid sample**, including quiet samples. The fresh panel covers the first and last generated shards, startup and 30-frame history, crying, whistling and a partial tail. This closes the numerical target-consistency question on the sampled cases, not every fresh training source.

The full teacher stage 4 output fed through the student's frozen suffix reproduced the original teacher waveform **bitwise exactly** for all 18 development/checkpoint combinations. The shared prefix output was also exact. Therefore the measured errors are introduced by the changed group rather than an independently broken or mutated suffix.

The pinned original implementation retains causal padding, the same transposed-convolution trim, Snake, residual skips and final convolution plus tanh. The compression code selects ConvTranspose input/output axes 0/1, ordinary convolution output/input axes 0/1, depthwise channels on axis 0, Snake channels on axis 1, and conditioning embedding input channels on axis 1. It reconstructs weight-normalization scale from the selected effective matrix. This audit found no incompatible axis, stride, dilation, padding or conditioning change. All measured effective weights and normalization parameters were finite. No zero normalization scale was found. The smallest absolute Snake alpha, 8.0469e-5, occurs in the original teacher too; its presence is not evidence of a new singular activation bug.

This is evidence against those concrete implementation faults. It does not establish that halving internal widths can reproduce every original nonlinear behavior.

## What intermediate measurements mean

The new trace includes each stage's conditioning output, input Snake, upsampler, and all three residual units' inputs, outputs and residual contributions, plus the final head. Stage 2/3 comparisons use selected teacher coordinates, with that limitation recorded. Their internal representation can legitimately change during joint group training. Large internal coordinate error does not prove a broken layer or identify a safe layer to delete.

The complete stage 4 boundary and frozen suffix share all coordinates. Measurements use exact teacher-defined near-silence windows, other quiet windows and active windows. Hidden DC statistics mean each channel's mean error within a region; they are not waveform offsets or a direct measure of acoustic loudness.

## Near-silence findings

| Source and metric | Step 1,000 | Step 4,500 | Step 5,000 |
|---|---:|---:|---:|
| Spanish near-silence failures |2/44 |44/44 |44/44 |
| Spanish near-silence residual RMS |9.402e-6 |8.624e-6 |7.165e-6 |
| Spanish student RMS |1.086e-5 |1.725e-5 |1.570e-5 |
| Spanish mean residual |−3.395e-6 |+6.762e-6 |+5.331e-6 |
| Kannada near-silence failures |20/42 |42/42 |42/42 |
| Kannada near-silence residual RMS |1.747e-5 |1.869e-5 |1.808e-5 |
| Kannada student RMS |1.838e-5 |2.425e-5 |2.319e-5 |

Teacher RMS is approximately 9.62e-6 in both sources. At 5,000, 43 of 44 Spanish failures and 41 of 42 Kannada failures are **amplitude-only**: their reconstruction residual already meets the existing absolute floor. The typical residual improves, but its mean shifts upward and output RMS crosses the tighter amplitude ceiling. These small absolute levels must not be described as proven audible noise. The checks remain provisional engineering thresholds, not calibrated audibility limits.

Kannada also has a distinct first 20 ms window with student RMS 1.095e-4 and residual RMS 1.0765e-4, versus 9.8354e-5 residual at 1,000. This startup energy error is larger than the stationary offset. It is separate from the previously observed teacher transient near 29.46 ms that the student fails to reproduce. Spanish's recorded near-silence occurs in an interior crop with 30 frames of context, so startup padding cannot explain its persistent offset.

Near-silence hidden errors decrease through much of the shared group and suffix. For Spanish, stage 4 error RMS falls .22015→.13050 from 1,000→5,000; head-Snake error falls .01215→.00964. Nevertheless the final mean residual changes sign. Teacher hidden features are not zero during acoustic silence: its head-Snake RMS is .27537 while waveform RMS is 9.62e-6. Small changes in feature combinations reaching the final projection can therefore matter at the output noise floor even while broad feature MSE improves. This observation does not isolate a particular trainable unit as the cause.

## Active amplitude drift

The later amplitude change is observed on ordinary speech too:

| Source | Stage 4 RMS, 4,500→5,000 | Stage 6 RMS, 4,500→5,000 | Waveform RMS, 4,500→5,000 |
|---|---:|---:|---:|
| Spanish |.35509→.36240 |.84344→.86190 |.038874→.043613 |
| Kannada |.32721→.33252 |.95897→.99394 |.057455→.062592 |
| English |.33546→.33996 |.98479→1.01821 |.058949→.063034 |

Teacher waveform RMS is .038858, .055180 and .054924 respectively. The original suffix propagates a changed group representation into a larger waveform-level change. Spanish group reconstruction error actually decreases over this interval, showing why its aggregate MSE alone cannot certify final amplitude.

A diagnostic intervention replaced the 5,000 group output with teacher output plus half the student error vector. With the suffix unchanged, active residual RMS decreased from .008508→.004062 for Spanish and .010285→.005061 for Kannada. Waveform gain relative to teacher decreased 1.122→1.059 and 1.134→1.070. Full teacher substitution gives exact output. This directly links the waveform problem to the group-output error supplied to an intact suffix. It does not prove the current training objectives give appropriate pressure to every sensitive direction.

## Decision

No tested pruning-copy, causal-geometry, frozen-suffix or target-consistency bug warrants rewriting the architecture or declaring earlier training invalid. The next decision should use the separately authorized gradient/objective audit to explain why group updates permit these output changes. Further residual-unit deletion is not supported by this diagnosis. Preserve the known checkpoints and keep the current nine-unit group while that question is resolved.
