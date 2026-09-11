# Step 4,000 review

The user reaffirmed continuing the unchanged current cut to 5,000, then requested an independent unpruned exact-teacher control. No training intervention was made.

Checkpoint SHA256: `857605e1be3c50dea428b2713bcd4904d17ca70748f333d13513218a91fb513c`.

All 15 integrity checks pass: checkpoint and receipt agree, all 90 Adam counters are at 4,000 with unchanged settings, frozen-state preservation is recorded, weights and losses are finite, the exact source order contains 48,000 distinct recordings, and all 24,000 new teacher/cache comparisons pass the existing tolerance. Passing that tolerance does not establish exact low-amplitude equality; the new control will examine this separately.

Compared with 3,500, waveform MAE worsens 3.22%, while mel error improves 3.72%, group MSE improves 6.22% and aggregate quiet residual RMS improves 4.27%. Mean active waveform cosine changes from 0.994505 to 0.994044. Forty-five of 96 recordings improve waveform MAE. The maximum absolute sample is 0.983852, with zero full-scale overshoot samples.

The new sustained/interior silence-level failures at 3,500 recover. Their window identities are unchanged.

| Cohort | Passing at 3,500 | Passing at 4,000 |
|---|---:|---:|
| All quiet | 1103/2544 | 1367/2544 |
| Near-silence | 1/184 | 171/184 |
| Sustained source silence after 40 ms | 0/164 | 164/164 |
| Interior near-silence after 800 ms | 0/50 | 50/50 |
| Startup first 20 ms | 0/13 | 0/13 |
| Teacher source-silence transient at 20 to 40 ms | 10/10 | 0/10 |

Sustained residual RMS falls from 3.134 to 1.402 millionths of full scale; output RMS falls from 12.589 to 8.810, compared with teacher 9.597. Interior residual RMS falls from 3.392 to 1.523, and output RMS from 12.718 to 8.874, compared with teacher 9.635. Both amplitude-only failure groups now pass completely. Their centered residuals are approximately unchanged, so the prior window-offset contribution has reduced.

Startup remains unresolved. Its residual RMS improves slightly from 20.831 to 20.612 millionths of full scale. The 20 to 40 ms teacher transient moves back outside the residual limit: RMS residual rises from 81.544 to 92.699 millionths of full scale, with ten residual-only failures. These opposing movements reinforce the need to retain continuous measurements and separate cohorts, rather than treat total correlation or a pass count as uniform recovery.

Continue unchanged to 5,000 as requested, with the scheduled 4,500 review retained. The post-run identical-teacher control is described in `../identical-teacher-control-v1/plan.md`; it will test copy/native inference, the current evaluator and caches, then a bounded optimizer stability probe. No new pruning cut is authorized.
