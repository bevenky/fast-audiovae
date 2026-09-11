# AudioVAE2 compression audit at 5,000 steps

The run completed normally. It is stopped at the requested review point, and the conditional continuation is paused. The objective is to validate the compressed architecture and training procedure, not to select a checkpoint that happens to score well. A demonstrated training bug should be fixed and the continuation replayed from the preserved 1,000-step checkpoint, subject to verifying that the bug did not already affect that checkpoint.

## What has been verified

- The original encoder and decoder teacher remain the references. The student is a copy of the original AudioVAE2 decoder with narrower internal channels in stages 2–4. This is not the earlier ConvNeXt student.
- Input to the replacement group is the same complete 1,024-channel tensor. Its output retains the complete 128-channel teacher interface. Frozen stages 5–6 and the original convolution/tanh output head remain differentiable with respect to the student's features.
- The source audit preserves convolution axes, effective weight normalization, selected Snake parameters, sample-rate conditioning, residual skips, dilation, causal padding and transpose-convolution trimming. All nine residual units in stages 2–4 are retained.
- The optimizer ledger reaches 5,000 for every trainable parameter. The 15,000 source IDs, audio hashes and parent recording IDs are unique. All 40 fresh-data shards passed receipt, hash, geometry and finite-value checks. No missing-gradient, nonfinite-loss or runtime error was found.
- The 96-source development panel and its target energy and masks are unchanged across checkpoints. It contains 94 active recordings, 2,544 quiet windows and 184 near-silence windows. All 22 scheduled Indic language codes are represented, usually by only one recording each.

## What is improving and what is regressing

| Metric | Step 1,000 | Step 4,500 | Step 5,000 |
|---|---:|---:|---:|
| Active waveform cosine | 94.47% | 97.19% | 97.25% |
| Waveform MAE, lower is better | 0.006786 | 0.004892 | 0.005530 |
| Whole stages 2–4 output MSE | 0.024986 | 0.011808 | 0.011695 |
| Quiet residual RMS | 0.0002442 | 0.0001814 | 0.0001769 |
| Quiet windows passing | 115/2,544 | 104/2,544 | 118/2,544 |
| Near-silence windows passing | 91/184 | 3/184 | 4/184 |
| Samples exceeding full scale | 0 | 0 | 0 |

From 4,500 to 5,000, MAE worsens by 13.04% on 78 of 96 recordings; MSE worsens by 10.21%. Pooled active output RMS relative to the teacher rises from 1.02093 to 1.08102. The number of active recordings louder than their teacher target rises from 52/94 to 83/94. Linear mel error worsens by 19.45%, while log mel improves by 2.61%. Their combined loss conceals that distinction.

The complete held-out weighted objective also worsens by 12.06%, from 0.00523582 to 0.00586722. This is not an improvement under the complete intended objective. A per-recording orthogonal decomposition attributes 67.18% of the late MSE increase to teacher-aligned amplitude error and the remaining increase to residual components orthogonal to the teacher waveform. This is a mathematical error decomposition, not evidence explaining why the optimizer moved in that direction. No gain correction was applied to the audio.

Cosine is invariant to positive amplitude scaling. Its 0.99 target is a waveform reconstruction goal, not a claim of 99% perceptual accuracy. Lower aggregate quiet RMS also does not imply that every near-silent window has improved. The quiet thresholds are provisional engineering checks, not calibrated audibility thresholds.

## Missing internal observations

The existing boundary observer records whole stages after their third residual unit. It does not record each upsampler, the first two residual-unit outputs, or separate residual contributions. Stage 2 and 3 comparisons use only the retained teacher coordinates, 256/512 and 128/256 respectively. These are partial diagnostics; the narrower representations can adapt, so internal equality is not a valid requirement. The full stage 4 output and later shared boundaries are valid comparisons.

The original 12-recording layer panel covers only 36 of the 184 near-silence windows. It misses the two largest regressions: Latin American Spanish loses 42 passing windows and Kannada loses 22. Those two recordings account for 64 of the net 87 additional failures. Improving aggregate boundary errors on that panel cannot establish that the problematic intermediate behavior is correct.

A targeted forward-only diagnosis has now measured six development recordings at checkpoints 1,000, 4,500 and 5,000, including the two missed near-silence failures. It records conditioning, upsamplers, each of the three residual-unit outputs and residual branches, the complete group output and the frozen suffix. Teacher features passed through the student's frozen suffix reproduce the teacher waveform bitwise on every tested checkpoint and source. No original teacher or student parameters were changed.

The near-silence pass collapse needs careful interpretation. At 5,000, 43/44 Spanish and 41/42 Kannada failures violate only the amplitude ceiling; the absolute teacher-residual test passes. The stationary mean error is approximately +0.000006. For Spanish, the pooled near-silence residual actually improves from 0.000009402 to 0.000007165 between 1,000 and 5,000, while student RMS rises from 0.00001086 to 0.00001570 against teacher RMS 0.000009624. The count drop therefore does not establish a large audible noise regression. The thresholds remain unchanged.

Kannada's first 20 ms is a separate startup failure: student RMS is 0.0001095, versus teacher RMS approximately 0.00001, with residual 0.0001077. This differs from its later missed teacher transient and from the persistent tiny offset in the interior Spanish crop. Source context and startup cases must be assessed separately.

The active amplitude change is larger. Spanish stage-4 feature RMS rises 2.06% from 4,500 to 5,000, while final active waveform RMS rises 12.19%. English shows 1.34% and 6.93% respectively; Kannada shows 1.62% and 8.94%. These are different representations with different units, so they are not operator-norm estimates. A diagnostic intervention halving the stage-4 feature-error vector lowers active residual RMS from 0.008508 to 0.004062 for Spanish and 0.010285 to 0.005061 for Kannada. This localizes the harmful input to the compressed group and its propagation through the intact suffix; it does not identify a single residual unit to delete or a proven training remedy.

## Data and target checks

Quiet audio is present: the fresh continuation contains 4,429 seconds of teacher targets with RMS at or below 0.001, including 222 seconds at or below 0.00001. Missing silence is not an explanation supported by the data.

Specific expressive coverage is thin. The complete run includes approximately 10 scored seconds from five whistling-labelled recordings and 31 seconds from 12 crying-labelled recordings. These are source labels, not verified event timestamps inside every crop. Broad expressive dataset membership must not be presented as actual event coverage.

Cached fresh targets come from full-source teacher inference, then cropping. Training computes hidden teacher features from the same cached latent crop with up to 30 past latent frames; the original decoder needs at most 20. Geometry and masks agree. The fresh producer checked repeated full-source decoding, but did not individually compare every cropped teacher decode with its cached waveform. The targeted audit now finds bitwise equality on six representative fresh sources covering first/last new shards, quiet, active, startup, interior, crying, whistling and a partial tail. The six development sources also match bitwise. This supports consistency on the inspected cases, not an inference check of all 12,000 fresh crops.

## Pruning decision

| Architecture | Estimated decoder GMAC per audio second | Reduction from original |
|---|---:|---:|
| Original teacher widths, nine group residual units | 8.9912 | 0% |
| Current narrower group, nine units | 5.1741 | 42.45% |
| Narrower group with six units | 4.7823 | 46.81% |

Removing another three units would save an additional 7.57% of the current candidate's estimated MACs and shorten temporal support. This does not establish its CPU RTF or quality. Preserve the original teacher unchanged; any additional pruning belongs in a separate student candidate. Do not combine a depth change with a training fix before attributing the current regressions.

## Gradient and optimizer diagnosis

A second bounded, no-update diagnostic tested the two near-silence failures, the largest late speech-amplitude contributor and the held-out whistling recording at 1,000, 4,500 and 5,000. Every source MAE reproduced its saved measurement. Teacher, frozen decoder, checkpoint bytes, gradient slots, module modes and RNG preservation passed. The algebraic AdamW direction was checked against an actual toy optimizer in two focused CPU tests, which also passed remotely.

The step-5,000 pooled parameter-gradient norms, using the unchanged training coefficients, were:

| Objective | Weighted gradient norm |
|---|---:|
| Waveform L1 | 0.385755 |
| Log mel | 0.001936 |
| Full stage-4 feature MSE | 0.001385 |
| Linear mel | 0.000218 |

Total-gradient cosine with waveform gradient was 0.999983. On these tested cases, mel and feature supervision are not overwhelming the waveform objective. Both the raw gradient and a hypothetical next AdamW direction on this panel would reduce active gain error in all three speech cases at 5,000. Broadly reducing mel/feature weights is therefore not supported by this probe.

Local conflicts do exist. For whistling, waveform and log-mel gradients have cosine -0.304 at 5,000. The hypothetical saved-momentum AdamW direction worsens its waveform and gain errors; zeroing only the first moment in the calculation reverses those signs. For the two near-silence cases, the raw gradient improves near-waveform error, while the saved-momentum direction worsens it; the same first-moment control reverses that effect. No moments were actually reset and no model update was made.

These controls establish local sensitivity to optimizer history on the selected recordings. They do not prove that resetting AdamW will improve the real run, and they do not reproduce the actual intervening fitting batches. In particular, near-silence residual improved slightly during the last 500 real updates despite these endpoint directions.

The actual 4,500-to-5,000 parameter displacement points toward higher speech-gain error under both endpoint gradients. Its contributions span convolution parameters in all three compressed stages; this is not evidence identifying a single removable layer. Endpoint directional derivatives are local approximations, not causal attribution of the complete finite training interval.

## Root-cause list and next decision

1. **Broad speech-amplitude drift: still unresolved at the training-process level.** Its numerical effect is measured and localized to the compressed group's changing representation. Tested target mismatches, frozen-suffix mutations and overwhelming mel/feature gradients have been ruled out on the inspected cases. Actual minibatch composition, small-batch variability and optimizer history remain candidates, not established causes.
2. **Stationary near-silence: measured small mean-offset and strict amplitude-ceiling interaction.** This differs from a large audible noise claim. Local optimizer-history sensitivity exists; a reliable training remedy has not been demonstrated.
3. **Startup and transient reconstruction: separate remaining approximation errors.** Correct causal geometry does not guarantee that narrowed channels retain the original transient mapping.
4. **Internal monitoring: a confirmed coverage gap.** The original subset missed the two main near-silence regressions and did not expose individual residual units. The targeted audit now measures those cases. Future training monitoring should include those actual failure regions and continuous residual/amplitude components, rather than only aggregate pass counts.
5. **Expressive exposure: confirmed sparse whistling and crying coverage.** Labels do not establish event occupancy in the selected crops. This should be addressed in the final recipe without pretending it explains the broad speech-amplitude drift.

The next diagnostic, if pursued, should replay the original 4,500-to-5,000 interval from its exact checkpoint, optimizer, RNG and source sequence in an isolated directory. First reproduce the saved endpoint; then inspect actual update directions and minibatch composition around the onset of gain drift. This is a proposed diagnostic, not a started training run. Do not select a favorable checkpoint, change loss weights, reset moments or delete layers as a substitute for identifying the mechanism.

No training-code defect has been demonstrated that justifies restarting from 1,000 with a claimed fix. The 1,000 checkpoint remains the agreed restart anchor once an appropriate correction is established. Current training and the conditional continuation remain paused.

## Evidence

- [Runtime, optimizer and data audit](runtime.md)
- [Matched validation and per-recording audit](validation-audit.md)
- [Full runtime evidence](runtime.json)
- [Full validation evidence](validation-audit.json)
- [Pruning and target geometry source audit](fresh-target-consistency-source-audit.md)
- [Layer and fresh-target numerical evidence](regressing-layer-diagnosis-v1.json)
- [Amplitude error decomposition](amplitude-error-decomposition.json)
- [No-update gradient and optimizer evidence](gradient-audit5000-v1.json)
- [Gradient diagnosis and interpretation](gradient.md)

No training restart, model promotion or additional layer deletion was performed for this audit.
