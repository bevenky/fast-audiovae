# GRAIL-like hidden-feature compensation: completed result

The independent G trial completed 2,000 updates from original teacher weights plus its declared native hidden-feature fits, with original RNG and fresh Adam. It saved its final checkpoint and passed the CPU-only audit: all 90 optimizer counters and finite states, initialization parity, protected files, frozen stages, checkpoint receipts, source ledger and saved quiet summaries were verified. All 24,000 sources were distinct and matched B's source order, representing 16.3336 hours of scored audio. All teacher-cache comparisons passed the original tolerance; one was not bitwise exact.

G is a promising retained candidate, but it does not resolve startup or transient reconstruction. No candidate is promoted and no additional training is authorized by this result. The independently initialized quiet-preservation trial has launched with its unchanged method.

| Metric at 2,000 updates | B | G |
|---|---:|---:|
| Waveform MAE, lower is better | 0.00261465 | 0.00247640 |
| Waveform MSE, lower is better | 0.0000769228 | 0.0000655914 |
| Mel error, lower is better | 0.141059 | 0.140855 |
| Stage 2–4 output MSE, lower is better | 0.00148276 | 0.00150627 |
| Mean active waveform cosine, higher is better | 0.992375 | 0.992986 |
| Active output RMS / teacher RMS | 0.999886 | 0.991455 |
| Aggregate quiet residual RMS, millionths of full scale | 78.734 | 77.740 |
| Quiet windows passing | 1,242 / 2,544 | 1,419 / 2,544 |
| Startup windows passing | 0 / 13 | 0 / 13 |
| Peak absolute sample | 0.988997 | 0.987242 |
| Samples above full scale | 0 | 0 |

G improved waveform MAE by 5.29% and MSE by 14.73%. MAE improved on 80/96 recordings, compared with 58/96 for downstream selection + B. Mel error was almost unchanged at 0.14% lower, with only 43/96 individual recordings improving. Internal group MSE worsened by 1.59%, including 67/96 individual recordings. Better waveform output alongside worse internal MSE is evidence against treating internal feature error as a complete quality measure.

| Quiet cohort | B passing | G passing | B residual RMS | G residual RMS |
|---|---:|---:|---:|---:|
| All quiet | 1,242 / 2,544 | 1,419 / 2,544 | 78.734 | 77.740 |
| Near-silence | 136 / 184 | 170 / 184 | 10.340 | 3.706 |
| Startup, first 20 ms | 0 / 13 | 0 / 13 | 38.457 | 13.163 |
| Teacher transient, 20–40 ms | 10 / 10 | 0 / 10 | 59.476 | 142.463 |
| Source-zero after 40 ms | 131 / 164 | 164 / 164 | 1.260 | 0.977 |
| Near-silence after 800 ms | 25 / 50 | 49 / 50 | 1.863 | 1.315 |
| Quiet with nonzero reference | 1,101 / 2,359 | 1,255 / 2,359 | 81.621 | 80.182 |

RMS values are millionths of full scale. Cohorts overlap and must not be summed. The disjoint split is startup 0/13, other near-silence 170/171, and remaining quiet 1,249/2,360. Lower startup RMS, by 65.77%, is not a startup pass. All 13 still fail both residual and amplitude limits. The 20–40 ms transient remains substantially worse than B. Aggregate quiet output-limit excess also improved, from 7.629 to 4.267 millionths of full scale, while 1,125 quiet windows still fail at least one criterion.

Relative to downstream selection + B, G has lower endpoint waveform and mel errors and more quiet passes, but worse internal group MSE. It is not uniformly superior across all individual recordings or quiet cohorts. All six review points remain in the report; no earlier checkpoint was substituted for the prescribed endpoint. This reused development panel is not an independent final test set, and cosine similarity is not perceptual accuracy.

Training-update compute took 1,193.39 seconds versus B's 1,201.59 seconds at matched exposure. This is training time, not inference RTF. The maps are folded into existing native operators, but CPU runtime and perceptual quality were not newly benchmarked.

- [Filtered aggregate audit](completed-aggregate.json).
- Final checkpoint SHA256: `542f0e8855ee5283ad9f04b00dc02ae7c720f2af31cdaff1183c9c43b8a1b23f`.
- Final checkpoint: `/dev/shm/fast-audiovae-grail-recovery-20260911-v1/results/checkpoint-step2000.pt`.
- Collector source SHA256: `a4d27c0fbf155372db1b2f8a3777b81b4bbddb7c157647db1ed28756f1fa556a`.
- CPU audit log SHA256: `1fbb757478650b7a2c9812685e3ab00b37c10e8800536cc44483854edabd5c76`.

Only aggregate reports were retrieved. Audio, latents, per-recording values and model weights remain on Runpod. No further inference, tests, architecture changes, optimizer changes or benchmarks were performed for this completion audit.
