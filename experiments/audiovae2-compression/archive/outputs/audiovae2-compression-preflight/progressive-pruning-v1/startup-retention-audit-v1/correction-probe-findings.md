# First rejected update repaired without changing the decoder

11 September 2026. The diagnostic completed in 218.02 seconds. It matched every recorded non-timing scalar through update 56, using the original 672-source prefix of the authenticated 768-source pilot stream. All initial-state, teacher, optimizer, RNG, original gradient-slot, frozen-parameter and protected-file checks passed. No checkpoint or counterfactual optimizer step was retained.

## Same-state comparison

| At the first rejected all-row update | Unchanged extended-grid result | Bounded current-trial correction |
|---|---:|---:|
| Actual weight movement, L2 | 0 | 0.01568781 |
| Actual movement / original Adam proposal norm | 0 | 0.9999786 |
| Calibration startup passing | 6/6 | 6/6 |
| Development startup passing | 12/13 | 12/13 |
| Same-batch ordinary total objective | 0.00542362 | 0.00520728 |
| Same-batch waveform MAE | 0.00523122 | 0.00501627 |
| Same-batch mel error | 0.249248 | 0.249860 |
| Same-batch group MSE | 0.00279959 | 0.00260729 |
| Development startup residual RMS, microFS | 3.65045 | 3.62060 |

The correction improves the same-batch total objective by 3.99%, although that batch's mel component slightly worsens. Development is observation-only and remains 12/13; this is not a solution for every unseen startup window.

One signed-bound solve was sufficient. Scale 1 reduced the largest positive relative excess from 0.00102496 to 0.00002839 but still failed one calibration window. Scale 2 passed all six. Its actual normal correction norm is 3.96486e-6, only 0.02527% of the original projected displacement norm and far below the fixed 25% cap. No fallback or second solve was needed.

## What this isolates

At this exact state, every positive fraction of the original projected direction, down through 1/1024, fails one calibration amplitude bound. Zero passes. Same-prediction FP64 scoring gives the same verdicts, so final metric reduction is not responsible for those rejections.

The ideal full linear prediction is at the boundary to numerical solver precision. Actual FP32 parameter writeback produces a linear excess of 3.024e-16; the actual decoder excess is 1.159e-13. The additional response beyond the linear prediction is 1.156e-13, much larger than parameter-writeback linear error. That remainder includes nonlinear response and forward arithmetic; this diagnostic does not uniquely separate them.

After recomputing the Jacobian at the rejected trial, the ideal first normal correction lands on the linear boundary. Native writeback and the remaining nonlinear response leave a small positive excess. The doubled normal moves inside the unchanged feasible limits. No threshold was relaxed and no output switch, bias-only patch or inference layer was added.

The complete signed solve resolves all twelve rows, verifies full KKT and reconstructed primal conditions, and has a rank-one active system with condition 1. This is evidence for an avoidable update-geometry failure at this state, not proof that the pruned architecture lacks capacity.

## Next qualification

Run a separate fresh 64-update pilot with this bounded correction policy from update one. Use the same C initialization, original RNG, empty Adam, original losses, 768 distinct ordinary sources and six separately counted recurring calibration anchors. Compare with all-row retention and ordinary recovery on sustained actual movement and the full 96-source panel. A successful one-state repair does not establish long-run recovery or qualify a GRAIL combination by itself.

All earlier implementations, results and failures remain preserved. [Aggregate](correction-probe-v1-aggregate.json), [protocol](correction-probe-plan.md), [execution](correction-probe-execution.md).
