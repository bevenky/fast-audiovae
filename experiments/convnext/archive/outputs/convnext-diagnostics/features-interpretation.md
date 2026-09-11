# Feature and output-head diagnosis

The current evidence does not identify a collapsed output head. The failure also extends beyond a repeating silence template. These are diagnostics of preserved checkpoints, with no model or optimizer updates.

The complete component audit now includes all 285 held-out crops, representing 144 natural sources and three synthetic fixtures. The previous four-row diagnostic was accidentally restricted by case-sensitive condition matching. The primary quality metrics were not affected by that selection bug.

| Checkpoint | Output matrix | Numerical rank | Condition number |
|---|---:|---:|---:|
| Parent, step 8,090 | 480 × 2,048 | 480 | 6.51 |
| Targeted, step 8,490 | 480 × 2,048 | 480 | 6.77 |
| Complex, step 8,490 | 480 × 2,048 | 480 | 6.91 |

The actual matrices have full row rank at both FP32 and FP64 thresholds. Their conditioning is moderate. This rules out a collapsed linear output basis, but cannot establish that the upstream network learns all coefficients the teacher needs.

## What remains in natural quiet audio

For the 1,409 complete quiet 40 ms cycles in crops with at least eight such cycles:

| Checkpoint | DC power share | Repeating 480-sample share | Additional 1,920-sample share | Remaining variation |
|---|---:|---:|---:|---:|
| Parent | 4.5% | 42.1% | 5.1% | 48.4% |
| Targeted | 14.7% | 17.2% | 6.9% | 61.2% |
| Complex | 12.9% | 20.2% | 6.9% | 60.0% |

Targeted training reduces absolute 480-periodic error power substantially, from 4.55e-8 to 1.42e-8. Remaining variation barely changes, from 5.24e-8 to 5.08e-8. Consequently, removing a fixed repeating template alone would leave a substantial residual. These component masks deliberately require complete cycles and exclude a full initial cycle for interior archived crops; their RMS must not replace the main 20 ms quiet metric.

Exact parent-to-candidate waveform-change attribution finds that changes in upstream features contribute about 42 times the head-weight-change power in natural quiet audio for targeted training, and 39 times for complex training. Both terms partly cancel. This localizes where the recent changes occurred; it does not prove which layer originally caused the error.

## Frozen-feature readout result

The probes used 1,024 training windows from 878 sources, split into 703 fitting sources and 175 tuning sources while keeping connected known speaker, session, recording and audio-hash identities together. Unknown speaker identities remain a limitation. No held-out source entered fitting or hyperparameter selection.

The initial grid ended at a ridge value of 0.01, and every selected probe reached that boundary. We therefore extended the same ridge family to 0.03, 0.1, 0.3, 1, 3 and 10, reproduced the original 0.01 measurement, and retained the identical fitting/tuning sources. All six probes now select 0.3, an interior point of the expanded grid. Held-out evaluation did not influence that selection.

The same-dimension bias-free probes show these changes relative to each checkpoint's existing head:

| Checkpoint | Fitting MSE | Tuning MSE | Held-out waveform MAE | Held-out mel error | Held-out quiet RMS |
|---|---:|---:|---:|---:|---:|
| Parent | -12.9% | +1.6% | +2.2% | -12.4% | +2.4% |
| Targeted | -14.4% | +1.6% | +2.0% | -11.9% | +13.9% |
| Complex | -14.3% | +1.4% | +1.4% | -10.0% | +9.1% |

Negative error changes are improvements. These probes improve mel fidelity, but do not consistently generalize the fitting MSE gain or preserve waveform and quiet fidelity. Adding an intercept produces similar tradeoffs. For targeted and complex checkpoints, overshoot sample counts fall 7.7% and 9.3% with the bias-free probe, but the maximum peaks still rise. A lower event count therefore does not establish that overshoot is fixed.

This closes the tested ridge-grid gap and finds no straightforward head-only correction that preserves all measured qualities. It does not prove that the current architecture cannot learn better upstream features. The fitting objective is squared waveform error with ridge regularization; it differs from the trained model's combined objectives. Probe capacity, finite fitting data and unknown speaker identities still limit the conclusion.

## Overshoot is not confined to block boundaries

After deduplicating overlapping crop observations by source and absolute sample index, overshoot affects 248 samples across six sources in the parent, 282 across five in targeted, and 283 across six in complex. None of the 37, 42 and 43 contiguous overshoot segments respectively has its peak within six samples of a 480-sample boundary. Most affected samples are concentrated in one Sindhi recording and one laughter recording. The pattern does not support treating every overshoot as a chunk-join error.

The linear decomposition agrees with actual FP32 output changes within 3.9e-6 at worst. Attribution powers include cancellation; they should not be reported as independent causal percentages. No deployment changes or quality-preserving solution have been established by this audit.
