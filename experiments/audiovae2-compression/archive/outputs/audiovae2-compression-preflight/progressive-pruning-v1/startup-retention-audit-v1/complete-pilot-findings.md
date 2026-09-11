# Complete startup constraints: fresh 64-update result

11 September 2026. The pilot completed in 235.42 seconds. All initialization, frozen-state, source-order, target-cache, optimizer/RNG restoration and source-file checks passed. Each independent pilot used the same fresh C initialization and 768 distinct ordinary sources. Retention additionally used six recurring calibration sources. No checkpoint was retained and no development input selected an update.

## Matched results

| After 64 updates | Ordinary Adam | Original two-max retention | Extended-grid two-max | All twelve constraints |
|---|---:|---:|---:|---:|
| Waveform MAE | 0.00488615 | 0.00542730 | 0.00541925 | 0.00503486 |
| Mel error | 0.233075 | 0.237675 | 0.237486 | 0.233467 |
| Stage-group MSE | 0.00277989 | 0.00304720 | 0.00304253 | 0.00274732 |
| Active waveform cosine | 0.974574 | 0.970705 | 0.970787 | 0.973271 |
| Calibration startup passing | 0/6 | 6/6 | 6/6 | 6/6 |
| Development startup passing | 1/13 | 12/13 | 11/13 | 12/13 |
| All quiet passing | 413/2,544 | 456/2,544 | 455/2,544 | 478/2,544 |
| Quiet residual RMS, microFS | 137.90 | 155.43 | 155.28 | 153.31 |
| Zero weight updates in final 16 | 0 | 16 | 15 | 4 |
| Full pilot seconds | 144.54 | 166.57 | 172.15 | 235.42 |

All twelve constraints reduce MAE 7.09% versus the extended-grid rule and 26.86% from initialization. Its MAE is still 3.04% worse than ordinary recovery. No arm overshoots full scale in this comparison. These are training-policy measurements, not inference speed or perceptual quality claims.

The fixed twelve-source fitting objective improves from 0.01053158 to 0.00711938, a 32.40% reduction. This is better than ordinary recovery's 0.00734255 but worse than extended-grid retention's 0.00689126, despite better full-development MAE. The fixed fitting batch is not the full development panel. Changing ordinary batches do not establish a late-stage loss trend.

## Learning is improved but still restricted near the boundary

The all-row pilot makes nonzero weight updates in 12 of its final 16 steps. Its first zero is update 56; the other zeros are 61, 62 and 64. In updates 49–56, accepted displacement norms sum to 0.008662 and mean fraction is 0.07043. In updates 57–64, those values fall to 0.001041 and 0.006958. Thus the new policy improves the earlier terminal stall, but the final interval still accepts less than 0.7% of the proposed movement on average.

All 64 linear projections pass the full reconstructed-row and KKT checks, and all six actual calibration starts pass after every accepted update. Full Gram matrices resolve to rank 12, with maximum condition 7.07e12. Accepted active systems have rank 0, 1 or 2, with maximum active condition 8.12. Maximum reported normalized primal violation is 6.19e-20. The large full condition alone does not prove a solver failure; the accepted active solutions and exact nonlinear acceptance must be considered together.

The mean update costs 2.035 seconds, including 0.595 seconds for the ordinary Adam work and 1.440 seconds for retention work. This adds training cost without changing the decoder graph or inference operations.

## Startup and other quiet cases remain distinct

| Development cohort passing | Ordinary | Original retention | Extended grid | All twelve |
|---|---:|---:|---:|---:|
| Near silence | 171/184 | 178/184 | 177/184 | 180/184 |
| First 20 ms startup | 1/13 | 12/13 | 11/13 | 12/13 |
| Teacher transient at 20–40 ms | 0/10 | 0/10 | 0/10 | 0/10 |
| Source-zero after 40 ms | 164/164 | 161/164 | 161/164 | 164/164 |
| Near silence after 800 ms | 49/50 | 45/50 | 45/50 | 47/50 |
| Quiet with nonzero reference | 248/2,359 | 284/2,359 | 284/2,359 | 303/2,359 |

Cohorts overlap. The remaining development startup failure is amplitude-only. The worst startup amplitude-squared excess is 3.673e-11, worse than extended-grid retention's 3.430e-11 despite a better pass count. Pooled startup residual RMS is 3.650 microFS. Passing all six calibration starts does not guarantee every unseen start or solve the later teacher transient.

## Decision

Preserve the all-row implementation and result as the best retention pilot so far. Before a fresh GRAIL combination or a full 2,000-update recovery, run the already proposed bounded correction at this policy's first zero, update 56. Reproduce its original fresh-state trajectory and exact non-timing scalar prefix, then compare the same Adam proposal with a current-trial Jacobian correction and unchanged actual startup limits. Keep data, architecture and objective fixed.

The correction is conditional on local evidence. No new inference layer, silence switch, loss weight, precision change or relaxed threshold is justified by this pilot. A later GRAIL hybrid must derive its own feasible startup initialization using its actual internal features and be qualified independently.

Sources: [complete aggregate](complete-anchor-pilot-v1-aggregate.json), [execution](complete-pilot-execution.md), [plan](complete-pilot-plan.md), [preceding same-state direction result](direction-probe-findings.md), [experiment register](../experiment-register.md).
