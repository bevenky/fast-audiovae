# Downstream selection plus B: completed 2,000-update result

The fresh repeat completed and saved its final checkpoint. The aggregate CPU audit passed for all 90 Adam parameter states, their counters and finite values, the fixed optimizer recipe, fresh initialization, checkpoint identities, protected files, source order and saved quiet-window summaries. It used 24,000 distinct training sources, 16.3336 hours of scored audio, and the same 96-source development panel as B. All 24,000 teacher-cache checks passed the original tolerance; one was not bitwise exact.

This is a mixed quality result, not a replacement for B. The same downstream support gave a better initial approximation, but most of the initial waveform advantage narrowed during recovery. The GRAIL and quiet-preservation arms continue under their original independent protocols.

| Metric at 2,000 updates | B | Downstream selection + B |
|---|---:|---:|
| Waveform MAE, lower is better | 0.00261465 | 0.00255562 |
| Waveform MSE, lower is better | 0.0000769228 | 0.0000697701 |
| Mel error, lower is better | 0.141059 | 0.141938 |
| Stage 2–4 output MSE, lower is better | 0.00148276 | 0.00127671 |
| Mean active waveform cosine, higher is better | 0.992375 | 0.991853 |
| Active output RMS / teacher RMS | 0.999886 | 0.989814 |
| Aggregate quiet residual RMS, millionths of full scale | 78.734 | 88.694 |
| Quiet windows passing | 1,242 / 2,544 | 1,158 / 2,544 |
| Startup windows passing | 0 / 13 | 0 / 13 |
| Peak absolute sample | 0.988997 | 0.988462 |
| Samples above full scale | 0 | 0 |

Waveform MAE improved 2.26%, waveform MSE 9.30%, and group MSE 13.90%. Mel error worsened 0.62%, and aggregate quiet residual RMS worsened 12.65%. MAE improved on 58 of 96 recordings and worsened on 38; mel improved on 42 and worsened on 54. Mean active cosine is a similarity metric, not a percentage of perceptual accuracy.

Quiet behavior is not uniformly worse. Near-silence passes improved from 136/184 to 165/184, and interior near-silence after 800 ms improved from 25/50 to 50/50. Startup RMS error improved from 38.457 to 26.109 millionths of full scale, but all 13 startup windows still failed. The teacher's 20–40 ms transient was reconstructed less accurately: residual RMS rose from 59.476 to 149.100 millionths of full scale and passes fell from 10/10 to 0/10. Nonzero-reference quiet passes fell from 1,101/2,359 to 998/2,359. These named cohorts overlap and must not be added together.

The startup success at initialization did not survive recovery: 11/13 passed at step 0, and 0/13 passed at every later review. Recovery was not monotonic; from 1,500 to 2,000, waveform MAE rose from 0.00227146 to 0.00255562. The prescribed 2,000-update endpoint remains the comparison point. No checkpoint was promoted and no extension was started.

The completed repeat exactly reproduced the preserved first attempt's source order, all four losses at all 2,000 updates, teacher-cache checks, and aggregate, per-recording and quiet summaries at all six saved reviews. This demonstrates reproducibility of the storage-only restart, not independent evidence across random seeds. The first attempt remains a failed run because it did not save its final checkpoint.

- Final checkpoint SHA256: `5b5885a6f4e50b41f35612fd8707d876ff56fa4cddb0c3cf184dc2e4e18a418b`.
- Final checkpoint remains on Runpod at `/dev/shm/fast-audiovae-downstream-recovery-20260911-v2/results/checkpoint-step2000.pt`.
- [Filtered aggregate audit](completed-repeat-aggregate.json) contains every milestone and cohort comparison.
- [Repeatability audit](repeatability-aggregate.json) contains equality counts and booleans only.

Training-update compute took 1,188.49 seconds. This is training timing, not CPU inference RTF. This review loaded saved reports and CPU checkpoints only; it performed no additional inference, training, benchmarks or tests. The reused development panel is not an untouched final evaluation, and no new perceptual or deployment claim is established.
