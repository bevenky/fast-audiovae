# Matched 64-update startup-retention pilots

Both pilots completed on 11 September 2026. They started independently from the exact same original-teacher-derived C initializer, original RNG and empty AdamW. Both used the same 768 distinct ordinary sources once, original losses and all 90 group parameters. The retention arm additionally used six fixed calibration starts as recurring constraints. All preservation checks passed; no trained pilot checkpoint was saved.

| Measurement | Ordinary control | Startup-only retention |
|---|---:|---:|
| Updates | 64 | 64 |
| Development startup passing | 1/13 | 12/13 |
| Calibration startup passing at endpoint | 0/6 | 6/6 |
| Waveform MAE | 0.00488615 | 0.00542730 |
| Mel error | 0.233075 | 0.237675 |
| Group MSE | 0.00277989 | 0.00304720 |
| Active waveform cosine | 0.974574 | 0.970705 |
| All quiet windows passing | 413/2,544 | 456/2,544 |
| Quiet residual RMS, microFS | 137.898 | 155.426 |
| Startup residual RMS, microFS | 5.03376 | 3.68336 |
| 20–40ms teacher-transient passing | 0/10 | 0/10 |
| Source-zero windows after 40ms passing | 164/164 | 161/164 |
| Near-silent windows after 800ms passing | 49/50 | 45/50 |
| Full-scale overshoot samples | 0 | 0 |
| Rejected parameter displacements | Not constrained | 21/64 |
| Total disposable pilot time | 144.54 s | 166.57 s |

Quiet cohorts overlap. The development panel has been reused across experiments, so it is not a final untouched test. Cosine is not perceptual accuracy. Total pilot time includes preparation, validation and final state/file checks; it is neither CPU inference RTF nor pure update time. Retention used39.12s of ordinary updates plus21.95s of constraint work.

Retention improved waveform MAE21.16% from its initial state, but its endpoint MAE is11.08% worse than the matched ordinary control. Its fixed first fitting batch improved too, so initial learning was real. However, its last16updates all had **zero parameter movement**:

| Update interval | Zero displacements | Mean accepted fraction |
|---|---:|---:|
| 1–16 | 0/16 | 0.9375 |
| 17–32 | 0/16 | 1.0000 |
| 33–48 | 5/16 | 0.3828 |
| 49–64 | 16/16 | 0.0000 |

Adam moments advanced once per ordinary batch as explicitly declared, even at zero accepted displacement. Those final steps are not successful reconstruction learning. All six training anchors remained valid at each accepted update, but one development start failed from the step32 review onward. This confirms the finite-anchor generalization limitation, while the widespread late zero steps expose an optimization limitation.

**Do not extend this implementation to2,000updates yet.** Preserve both pilots and diagnose the first zero displacement: selected versus omitted constraint gradients, curved-boundary effects, max switching, the fixed backtracking grid and actual FP32 writeback. Changing quiet thresholds or merely adding more training would not resolve a stalled displacement rule. The next diagnosis starts fresh and keeps this implementation immutable.

Evidence: [ordinary aggregates](ordinary-pilot-v2-aggregate.json), [retention aggregates](anchor-pilot-v2-aggregate.json), [preceding layer intervention](measured-findings.md), [method review](literature.md).
