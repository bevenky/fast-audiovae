# Native upsampler reconstruction: what explains the gap

The approved tests completed. They identify a substantial dependency between the trained upsampler and its downstream residual units. They do not produce a replacement that beats the preserved trained student. Keep the existing candidate and the main run paused.

These comparisons start from the retained step-4,625 candidate, the common starting point of the previous hint experiment. They do not start from either step-4,750 experimental endpoint. No neural training was performed here.

## The clearest controlled result

All rows use the same 96 development recordings and the trained upstream stages, except where the teacher directly supplies the intermediate output as a diagnostic oracle.

| Variant | Waveform MAE | Active correlation | Active RMS level relative to teacher | Quiet passes | Near-silence passes |
|---|---:|---:|---:|---:|---:|
| Preserved student | 0.00423436 | 97.6007% | 96.86% | 138 / 2,544 | 18 / 184 |
| Exact teacher upsampler output into adapted residual units | 0.0197830 | 99.3828% | 166.32% | 53 / 2,544 | 1 / 184 |
| Fitted upsampler into adapted residual units | 0.0203498 | 96.2999% | 160.76% | 0 / 2,544 | 0 / 184 |
| Same fitted upsampler into original teacher residual units | 0.00490318 | 97.0515% | 93.89% | 89 / 2,544 | 5 / 184 |

The exact teacher upsampler output makes waveform MAE 4.67 times worse when connected to the trained residual units. Those units have adapted to the pruned upstream representation. Replacing that representation with a teacher-correct intermediate signal does not remove their learned adjustments.

Holding the fitted upsampler fixed and changing only the three existing stage-4 residual units to the original teacher weights reduces MAE by 75.91%. This directly establishes that the downstream residual units mediate much of the isolated replacement's failure. It is more specific than inferring a cause from a loss curve alone.

However, that coordinated replacement still has 15.79% worse waveform MAE, 37.09% worse mel error and 9.35% worse quiet RMS than the preserved student. Waveform and mel error worsen on 95 of 96 recordings. It is not an acceptable repair.

The exact-teacher-feature oracle also demonstrates why 0.99 correlation cannot be our sole acceptance criterion: it exceeds that target while producing active audio at 166% of the teacher's RMS level. Every variant retains the original tanh head and has zero full-scale overshoot samples, so that count alone does not establish correct loudness either.

## Where reconstruction remains incomplete

The native fit preserves the same 128-input/128-output, stride-2, kernel-4 convolution, two output phases, causal history and one shared bias. It reduces the trained student's upsampler MSE by 81.91% overall and 94.91% in near-silence. Both phases improve; this is not a wrong-phase or missing-context implementation failure.

On the development panel, the fitted upsampler's MSE is:

| Inputs used for the local prediction | MSE |
|---|---:|
| Retained coordinates from the original teacher input | 0.000304315 |
| Actual trained student input | 0.00149656 |
| Actual untrained sliced student input | 0.00379348 |

The trained student's input gives 4.92 times the local fit error of retained original teacher inputs. Training has nonetheless improved this predictability substantially compared with the original sliced initialization. This locates another part of the approximation gap upstream of stage 4.

This comparison does not prove irreversible information loss: the representations differ, and these probes use a fixed linear operation. A nonlinear recovery or a different retained representation could behave differently. The full teacher prefix also performs computations absent from the narrowed student; its retained coordinates are an explanatory reference, not free student inputs.

The original sliced model with exact teacher upsampler output reproduces the teacher waveform to MAE 4.8e-9, with every quiet and near-silence window passing. Supplying the complete teacher stage-4-end output to the common frozen suffix is bitwise exact on all 96 sources. Thus the external stage-2-to-4 block contract and the frozen waveform suffix are valid.

## What this says about the hints

The original gap predates the hints. Narrowing the two internal channel boundaries changes what the remaining layers compute, and the trained layers compensate together. Correct shared latents ensure the same conditioning; they do not guarantee that a smaller decoder reproduces the full teacher function.

The previous projected hints genuinely improved feature predictability, but feature MSE does not weight all errors by their effect on the final nonlinear waveform path. Its fixed coefficients also did not maintain the calibration gradient fractions on every batch. These are relevant limitations, not a demonstrated optimizer or teacher-target bug.

The new oracle replaces raw intermediate activations, while the earlier hints were projected auxiliary losses and never entered the inference path. Consequently, the oracle proves internal coadaptation and the danger of isolated replacement. It does not uniquely establish the cause of the small earlier hint regression, nor fully separate capacity, initialization and optimization contributions to the remaining trained-model gap.

## Recommendation

Keep the current trained weights. Do not adopt the hints, isolated upsampler refit, or refit plus restored residuals from these results.

Use the complete stage-4 output and final waveform as the recovery targets, treating its upsampler and three residual units as one function. These targets already exist in our current whole stages-2-to-4 training; they are not missing losses that we need to add.

A useful next bounded comparison would freeze the current stages 2 and 3 temporarily and recover the complete stage-4 block from its stable actual inputs, starting from the preserved weights. Compare it against the unchanged joint-training recipe at identical exposure. This tests how much of the remaining error the final block can recover without its input representation changing at the same time. It would add no inference operators. This experiment has not been launched, and improvement is not guaranteed.

If recovery remains insufficient, the next architecture choice should revisit which channels or linear factors are retained. Do not remove further channels or assume that a longer run alone will repair the gap.

## Execution and evidence

All 10 focused tests passed locally and on Runpod. The diagnostic completed in 46.60 seconds after model loading. Original teacher, model states, checkpoint files and frozen stages were preserved. No new layers, CPU benchmark, neural training, commits or automatic promotion occurred.

A report filename collision overwrote the teacher-retained-input solver metadata with its calibration-error report. The numerical results were intact. The identical 72-source fit was rerun solely to recover the missing metadata in a separate file. Its normal-equation relative residual is 2.41e-15. The audit records 1,224 teacher/cache comparisons from the main diagnostic and 72 additional recovery comparisons, all bitwise equal; these counts are repeated checks, not distinct recordings.

![Waveform and level effects of the diagnostic substitutions](results/reconstruction-diagnostics.png)

The waveform intervals are selected from the saved teacher-defined near-silence and peak intervals. The teacher-feature oracle requires the original teacher and is not deployable. RMS level is the square root of pooled active student energy divided by pooled active teacher energy; it is distinct from a least-squares signed gain.

Supporting records: [protocol](protocol.md), [independent analysis](independent-analysis.md), [execution audit](execution-audit.md), [previous hint interpretation](../projected-hints-v1/method-result-review.md), [raw results](results/completed.json).
