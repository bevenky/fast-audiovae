# 400-step comparison

The quarter-rate run improves reconstruction and quiet noise substantially, but it **fails the two peak-control gates**. It is the stronger research checkpoint, not a release-ready replacement. Both final checkpoints and the original baseline are preserved; neither candidate was promoted.

Both runs started from step 8,490 and completed 400 updates using the same 12,800 freshly regenerated training windows, approximately 8.98 scored audio hours across 3,621 sources. Scored intervals did not repeat within either run. Initial optimizer moments, loss EMA, discriminator state and normalization were identical. The only difference between arms was the joint generator learning rate.

The canonical evaluation contains 282 natural crops from 144 sources, plus three synthetic fixtures. Natural-audio results are:

| Metric | Starting checkpoint | Current rate | Quarter rate |
|---|---:|---:|---:|
| Raw waveform MAE ↓ | 0.0125445 | 0.0122285 | **0.0116030** |
| Mel error ↓ | 1.13546 | 1.12742 | **1.04814** |
| Quiet residual RMS ↓ | 0.000306715 | 0.000315697 | **0.000258495** |
| Mean nonquiet correlation ↑ | 0.920111 | 0.924717 | **0.928423** |
| Maximum absolute amplitude ↓ | 1.265507 | **1.256053** | 1.296632 |
| Samples exceeding full scale ↓ | 457 | 502 | 588 |
| Quiet windows passing checks ↑ | 0 / 3,434 | 0 / 3,434 | **48 / 3,434** |

Against the starting checkpoint, the quarter-rate run reduces MAE by **7.51%**, mel error by **7.69%**, and quiet residual RMS by **15.72%**. Quiet RMS is **18.12% lower than the matched current-rate run**. These are measured reconstruction errors, not listening-quality scores.

The separate encoded-zero fixture improves substantially: steady-state residual RMS falls from **0.000137479 to 0.000043793**, a **68.15% reduction**. Nevertheless, all 200 steady-state quiet windows still fail the existing strict checks. Natural quiet failures also remain high at 3,386 of 3,434. The quality target has not been reached.

## What passed and failed

I independently recomputed all recorded gate comparisons. The quarter-rate candidate passes **90 of 92 gates in each evaluation domain**, historical and canonical. Both fail the same two requirements:

- Maximum natural peak rises **2.46%**, exceeding the permitted 1% increase.
- Mean peak-excess energy rises **53.72%**, exceeding the permitted 10% increase.

The largest peak is in the Sindhi recording `5629499534317154_chunk_1.flac`, reaching **1.296632**. A laughter source, `freesound:25794`, also rises from **1.167517 to 1.199306**. Lower average reconstruction error therefore has not prevented worse individual transients.

Every natural source improves in raw MAE and mel error: **144 of 144** for each metric. All 41 reported language/event groups improve on these averages, including the 39 groups large enough for gating. This describes this validation sample, not every speaker or utterance in those languages.

Quiet RMS improves in **84 of 87 sources** with qualifying quiet windows. Three regress: one Odia source by **39.50%** across six quiet windows, one Cantonese source by **10.37%** in one window, and the screaming source `freesound:220649` by **4.01%** in one window. Their waveform and mel averages still improve. These cases remain explicit exceptions to the aggregate quiet improvement.

## Integrity and streaming

The two starting evaluation reports match exactly in both domains, and their original engine hashes match. I separately read the durable discriminator-view logs on the server: their SHA-256 hashes are identical, confirming the same view schedule.

All **eight CPU streaming comparisons** pass: two inputs per candidate at 80 ms and 160 ms. I recomputed every chunk's expected sample count and its contribution to the complete output. No samples were dropped. Maximum waveform differences from batch execution were **6.56×10⁻⁷** for current rate and **9.83×10⁻⁷** for quarter rate, below the 2×10⁻⁶ tolerance. This verifies these streaming cases; no new CPU RTF benchmark was run.

Keep the quarter-rate checkpoint for the next focused investigation, with peak control as the remaining selection blocker. Do not relax the failed gates or treat the improved correlation as proof of teacher-level quality.

Evidence: [compact independent summary](400-step-summary.json), [canonical gate results](400-step-evidence/comparison-canonical.json), [historical gate results](400-step-evidence/comparison-historical.json), [experiment identity](400-step-evidence/identity.json), [preserved checkpoint hashes](400-step-evidence/checkpoint-preservation.json).
