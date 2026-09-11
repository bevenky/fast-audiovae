# Fresh G-plus-startup recovery: completed 2,000 updates

11 September 2026. The run completed and stopped at its declared 2,000-update limit. The CPU-only completion audit passed at 2026-09-11T12:46:58.885894+00:00. All prior experiments and checkpoints are preserved. This is a completed recovery experiment, not a promoted model or a new inference-speed benchmark.

The unchanged 384/256/128 stage widths retain all nine residual units and train 90 native tensors. The original encoder/teacher and outer student stages remain frozen. Exactly 24,000 distinct ordinary sources were used once, plus six separately counted recurring calibration starts (12,000 anchor update participations). AdamW remained continuous throughout.

| Metric | Step 1,500 | Step 2,000 |
|---|---:|---:|
| Active waveform cosine | 0.992913 | 0.992533 |
| Waveform MAE | 0.00245418 | 0.00256754 |
| Mel error | 0.146298 | 0.143115 |
| Group feature MSE | 0.00162577 | 0.00152453 |
| Active RMS / teacher | 1.003920 | 0.982034 |
| Protected calibration starts | 6/6 | 6/6 |
| All quiet windows passing | 1222/2544 | 1334/2544 |
| Development startup passing | 11/13 | 12/13 |
| Quiet nonzero-reference passing | 1039/2359 | 1215/2359 |
| Near-silence passing | 180/184 | 117/184 |
| Zero-source 20–40 ms transient passing | 10/10 | 10/10 |
| Zero-source after 40 ms passing | 162/164 | 98/164 |
| Near-silence after 800 ms passing | 49/50 | 4/50 |
| Quiet residual RMS, millionths full scale | 79.676 | 77.717 |
| Peak absolute amplitude | 0.985749 | 0.990278 |
| Overshoot samples | 0 | 0 |

The seven quiet cohorts overlap and must not be added together. Overall quiet passing improves to 1,334/2,544, but 1,210 still fail. All 67 near-silence failures and all 66 sustained-zero failures at the endpoint are amplitude-only under the existing limits. This states the measured failure category, not its root cause. The user has deferred that investigation until after the four optimizer comparisons.

Waveform MAE worsened 4.62% versus 1,500, while mel and group-feature errors improved. Active RMS is 98.203% of teacher. The dashboard's final whistling RMS ratio is 92.873% of teacher; no earlier ratio is inferred here. Correlation 0.992533 passes the 0.99 milestone, but is not 99% perceptual accuracy and does not establish silence quality.

## Integrity and runtime

All 0/1,000/2,000 checkpoint hashes and byte receipts match. Each contains 90 finite group tensors; optimizer state is empty at 0 and covers all 90 tensors at 1,000 and 2,000 with exact counters, finite moments, unchanged recipe, source ledger and RNG. All 57 protected files rehash correctly. Completion receipts report unchanged teacher, outer stages and runtime globals. The fixed development window identity and all saved full-review receipts match.

All 2,000 updates move weights and preserve the six calibration starts. There are 872 accepted nonlinear corrections, 883 normal solves and 3 fraction fallbacks, with zero zero-displacement updates. Post-setup elapsed time is 89.56 minutes: ordinary updates 19.47 minutes, protection auxiliary 67.74 minutes, full validation 23.12 seconds, with remaining overhead outside these components. These are training timings, not CPU decoder RTF.

Audit bookkeeping: the first audit incorrectly compared the checkpoint's full development aggregate with a filtered summary. The exact check against the original saved development aggregate passes at all three checkpoints. The first audit is preserved in completion-audit-v1-aggregate.json; the corrected audit is completion-audit-aggregate.json. No model, training output or quality threshold changed.

## Next sequence

Preserve this endpoint and stop here. The current model is not cleared for a deeper cut or release. The AdamW/Muon/NorMuon/Shampoo comparison remains planned; the alternative optimizer arms have not run. Keep the identical initialization, source exposure, existing protection policy and fixed quality panel for qualified comparisons. Perform the dedicated quiet-pass investigation after all four comparisons, as the user requested. No further training, inference benchmark or diagnostic started during this completion review.

The completed-run heartbeat is paused. TensorBoard and the pod remain available; all historical artifacts remain intact.

[Completed aggregate](completed-aggregate.json), [checkpoint audit](completion-audit-aggregate.json), [final review](review-step2000-aggregate.json), [optimizer plan](../muon-variants-review.md).
