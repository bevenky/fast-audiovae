# Complete startup constraints allow a larger useful update

11 September 2026. The two-arm comparison completed. The original 43-update non-timing scalar sequence was reproduced exactly, and all fresh-state, optimizer, RNG, teacher and file preservation checks passed. No counterfactual weights or checkpoint were saved.

## Same state, same Adam proposal

| Measurement | Before update | Two current maxima | All twelve individual constraints |
|---|---:|---:|---:|
| Fraction of projected proposal accepted | N/A | 1/32 | **1/2** |
| Actual parameter displacement L2 | 0 | 0.000546389 | **0.008741598** |
| Calibration startup passing | 6/6 | 6/6 | 6/6 |
| Development startup passing | 7/13 | 11/13 | **12/13** |
| Same-batch total ordinary objective | 0.00688955 | 0.00687846 | **0.00675063** |
| Same-batch waveform L1 | 0.00669539 | 0.00668445 | **0.00655829** |
| Development startup residual RMS, microFS | 3.67754 | 3.67848 | **3.65998** |

The new direction accepts 15.999 times the parameter movement and lowers the same-batch ordinary objective by 1.858% relative to the two-maximum direction, or 2.016% relative to the pre-update state. This is a local optimization result, not a CPU speedup or a proven training convergence rate. Neither arm used development measurements to choose its direction or accepted fraction.

The baseline is the state after update 42, which was not a review point in the earlier pilot. Its seven development passes do not contradict the earlier twelve passes at steps 32 and 64. Passing counts can vary near the limits, so the continuous output and residual measurements remain part of the aggregate.

The improvement is not uniform: pooled development startup residual RMS falls only 0.477%, and the worst development amplitude excess increases from 3.33582e-11 before the step to 3.44195e-11 afterward. The direction is more usable, but it has not repaired the remaining worst startup case.

## Why the larger constraint set helps here

The old projection protected the two windows with the largest current residual and amplitude excess. That subset missed another window endangered by the proposed update. The complete solver considered every individual inequality and found a different direction that remained very close to Adam's proposal: projection cosine 0.9999403, correction norm 0.0001910 versus proposal norm 0.0174844.

More constraints do not enlarge the linear feasible set. Here they improve the direction before nonlinear backtracking, allowing a much larger fraction to pass the actual decoder. Only one constraint was active in the final linear solution, but choosing it required considering the whole set. The active solve was well conditioned; the complete normalized Gram matrix remained nearly dependent and its reported condition number was about 3.52e12. Both facts are retained rather than calling the whole problem well conditioned.

The full all-twelve step still failed one actual amplitude check. Its ideal linear prediction was feasible, FP32 parameter writes produced a tiny positive linear excess of 1.47e-16, and the actual excess was 1.32e-13. The latter is much larger, so parameter rounding alone does not explain the rejection. Nonlinear response and forward arithmetic remain involved. Same-prediction FP64 scoring retained the failure. The half-step passed all twelve true inequalities with an amplitude margin of 4.72e-13.

## Decision

Test **complete-constraint projection alone** in a fresh 64-update recovery pilot with the same extended fraction grid. Keep the ordinary losses, AdamW, fresh C initialization, first 768 unique ordinary sources, six recurring calibration anchors and original limits identical. This policy changes from the first update, so require initial/data parity rather than falsely expecting the old 42-update scalar prefix.

Defer the nonlinear correction while testing this simpler method. The short pilot must show sustained nonzero learning, startup retention and reviewable ordinary/quiet quality. Success at this one state does not establish behavior throughout recovery or all thirteen development passes. A GRAIL combination remains conditional on that qualification and its own startup-feasible fresh initialization.

Evidence: [aggregate](direction-probe-v1-aggregate.json), [protocol](direction-probe-plan.md), [source and test receipts](direction-probe-execution.md).
