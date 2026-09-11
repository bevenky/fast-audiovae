# Why the startup-preservation update stopped moving

11 September 2026. The bounded replay completed through update 43. Every recorded non-timing update scalar matched the original 64-update pilot. The original model, empty optimizer, RNG, frozen teacher, outer layers and protected files were restored. This is scalar-prefix reproduction, not a saved full-tensor trajectory comparison. Only aggregate statistics were exported.

## Direct result

The old backtracking grid stopped at 1/16 of the proposed displacement before choosing zero. A valid, nonzero 1/32 step existed at that same state.

| Fraction of projected update | Linear inequalities passing | Actual startup anchors passing | Same predictions scored in FP64 |
|---|---:|---:|---:|
| 1 | 10/12 | 1/6 | 1/6 |
| 1/2 | 11/12 | 1/6 | 1/6 |
| 1/4 | 11/12 | 5/6 | 5/6 |
| 1/8 | 12/12 | 5/6 | 5/6 |
| 1/16, old smallest positive step | 12/12 | 5/6 | 5/6 |
| **1/32** | **12/12** | **6/6** | **6/6** |
| 1/64 | 12/12 | 6/6 | 6/6 |
| 1/128 | 11/12 | 6/6 | 6/6 |

These are the six training-calibration anchors, not the 13 development starts. Each anchor has separate residual and amplitude inequalities. At 1/32 the realized FP32 parameter displacement has L2 norm 0.000546389 and changes 6,547,680 parameter elements. It is not a rounded zero. No counterfactual displacement was retained as training.

## What is established

1. **The finite step grid caused an avoidable zero update.** Its smallest attempted positive step failed, but the next halving passed. This directly justifies extending the grid before adding another optimization method.
2. **The scalar linear model is insufficient at this state.** At both 1/8 and 1/16, every individual linear prediction passes but the actual decoder fails an amplitude bound. Evaluating the same prediction in FP64 does not change that verdict. Omitted constraints alone cannot explain rejection at those fractions.
3. **Omitted gradients matter for some larger proposals.** At 1/2 and 1/4 an omitted amplitude inequality is already linearly violated. The two-maximum projection therefore does not describe every potentially limiting window. A complete-row projection remains a distinct candidate if the simpler grid repair fails.
4. **Smaller is not monotonically safer in the actual FP32 network.** The tested 1/256, 1/512 and 1/1024 points pass only 5, 3 and 4 anchors, respectively. Ideal displacement, realized parameter writes and the executed nonlinear network must be distinguished. There is no justification for replacing actual checks with a derivative sign or assuming arbitrary halving will converge to an accepted representable update.

The nonlinear remainder includes network arithmetic as well as nonlinear response. It is not a measured pure Hessian term, and the same-prediction FP64 diagnostic does not convert the model itself to FP64. The direction reconstruction was exact. The all-row normalized Gram matrix had numerical rank 12 under its recorded tolerance, with a very small minimum eigenvalue; this is not proof that a larger constrained solve will be easy or well conditioned.

## Next bounded comparison

Run the same independent fresh-C 64-update pilot with only one change: append halvings from 1/32 through 1/1024 before the existing zero fallback. Keep Adam, original losses, all 90 trainable tensors, six anchors, precision, thresholds and the first 768 unique ordinary sources identical. Authenticate the non-timing prefix through update 42 against the original pilot. Restore all state afterward and save aggregate evidence only.

The purpose is to test whether the demonstrated escape sustains useful recovery. Compare terminal zero steps, accepted displacement, ordinary waveform and mel error, all quiet cohorts and development startup passing. A nonzero step at one state is not enough to authorize a 2,000-update version automatically.

If useful motion still stalls, the next targeted comparison is complete individual constraint projection, followed separately by a bounded normal correction based on measured rejected output excess. These are conditional tests, not changes bundled into this pilot. The primary-source reasoning and its limits are in [the nonlinear feasibility review](feasible-update-review.md).

Evidence: [first-zero aggregate](first-zero-v1-aggregate.json), [matched pilot comparison](pilot-findings.md), [initial layer interventions](measured-findings.md).
