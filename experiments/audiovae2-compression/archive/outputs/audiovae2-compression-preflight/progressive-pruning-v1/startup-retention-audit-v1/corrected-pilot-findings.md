# Corrected retention sustains learning in the fresh pilot

11 September 2026. The independent fresh 64-update pilot completed in 274.25 seconds. It used the exact original C initialization, RNG, empty Adam, 768-source ordinary stream, original objective and full 96-source evaluation. All initialization, cache, frozen-state, source, optimizer/RNG and runtime restoration checks passed. No trained diagnostic checkpoint was used or retained.

## Matched comparison

| After 64 updates | Ordinary recovery | All twelve linear constraints | Twelve constraints plus bounded correction |
|---|---:|---:|---:|
| Waveform MAE | 0.00488615 | 0.00503486 | 0.00488596 |
| Mel error | 0.233075 | 0.233467 | 0.233017 |
| Group MSE | 0.00277989 | 0.00274732 | 0.00277891 |
| Active waveform cosine | 0.974574 | 0.973271 | 0.974584 |
| Calibration startup passing | 0/6 | 6/6 | 6/6 |
| Development startup passing | 1/13 | 12/13 | 12/13 |
| All quiet windows passing | 413/2,544 | 478/2,544 | 466/2,544 |
| Quiet residual RMS, microFS | 137.90 | 153.31 | 137.80 |
| Zero parameter updates | 0 | 4 | 0 |
| Full pilot time, seconds | 144.54 | 235.42 | 274.25 |

The corrected rule matches ordinary recovery's waveform MAE to within 0.004%; this tiny difference is not evidence of a meaningful general-quality improvement. It preserves substantially more measured startup cases while removing the earlier late-update restriction. The fixed fitting objective is also effectively equal: 0.00734197 corrected versus 0.00734255 ordinary.

All 64 updates make real parameter changes and accept a base projected fraction of one. Twenty-eight updates use successful normal corrections; none falls back to the grid. The largest accepted normal is 0.07817% of its original projected displacement norm, well inside the unchanged 25% cap. In the final sixteen updates, actual displacement averages 99.9994% of Adam's proposal norm. A base fraction of one with a normal is not the unmodified Adam vector; the added normal and actual movement are recorded separately.

Every accepted update passes all six calibration windows and all twelve canonical inequalities. Every numerical projection and installed correction passes its recorded certificate and budget checks. This qualifies the update method for a fresh G-based short comparison. It does not establish 2,000-update convergence, every unseen startup case, final perceptual quality or CPU RTF.

## Quiet behavior is not uniformly better

| Development cohort passing | Ordinary | All twelve linear | With bounded correction |
|---|---:|---:|---:|
| Near silence | 171/184 | 180/184 | 182/184 |
| First 20 ms startup | 1/13 | 12/13 | 12/13 |
| Teacher transient at 20–40 ms | 0/10 | 0/10 | 0/10 |
| Source-zero after 40 ms | 164/164 | 164/164 | 164/164 |
| Near silence after 800 ms | 49/50 | 47/50 | 49/50 |
| Quiet with nonzero reference | 248/2,359 | 303/2,359 | 291/2,359 |

Cohorts overlap. Startup residual RMS is 4.167 microFS, higher than the all-row pilot's 3.650 and the common initializer's 1.956. The worst development startup amplitude-squared excess is 4.1265e-11. Near-silence RMS is 2.523 microFS despite its improved pass count. Hard acceptance constraints preserve bounds on the six training starts; they do not minimize every residual or guarantee unseen-window behavior. No full-scale overshoots occur.

The 20–40 ms transient remains unresolved and separately tracked. The original G2k result and this 64-update pilot are different exposure budgets and are not direct quality competitors.

## Training cost and next step

Average corrected update time is 2.635 seconds: 0.599 seconds ordinary Adam work and 2.036 seconds retention work. The measured accounting includes 248 all-six canonical score calls, or 1,488 source forwards, plus 552 source forwards for constraint gradients. Initial warmup and full development reviews are separate. These costs are training work; the exported decoder graph and inference operations remain unchanged.

Keep the corrected rule as the strongest qualified retention pilot. The approved next step is a fresh GRAIL-plus-constrained-upsample initializer using G's actual hidden features, followed by matched ordinary-versus-retention short pilots from that identical fresh state. Keep all three G residual mixers and refit only the existing native stage-3 upsampler. No trained retention or G endpoint is transplanted. [Combination plan](fresh-grail-combination-plan.md).

All historical candidates, source files, failures and aggregate evidence remain preserved. [Aggregate](corrected-anchor-pilot-v1-aggregate.json), [execution](corrected-pilot-execution.md), [local correction evidence](correction-probe-findings.md), [experiment register](../experiment-register.md).
