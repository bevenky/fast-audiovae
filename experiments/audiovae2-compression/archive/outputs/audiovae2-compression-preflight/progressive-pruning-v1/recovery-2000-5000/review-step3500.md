# Step 3,500 continuation review

The same student continues toward the approved 5,000-update limit. No architecture, optimizer, loss, data-order or pruning change was made for this review. This note is a summarized assessment of remote computations; raw reports and checkpoints remain on Runpod.

## Reconstruction

Compared with step 3,000, waveform MAE improves 7.29%, mel error improves 4.58%, group MSE improves 8.82%, and aggregate quiet residual RMS improves 13.26%. Mean active waveform cosine rises from 0.994272 to 0.994505. Waveform MAE improves on 77 of 96 recordings. All 96 recordings improve in waveform MAE, mel error and group MSE versus step 2,000.

The previous amplitude regressions recover: student RMS relative to teacher changes from 106.92% to 102.14% for the flagged Luganda recording, 104.17% to 100.04% for Maithili, 103.84% to 98.27% for Javanese and 104.00% to 100.35% for Portuguese. Whistle 428921 improves waveform MAE by 12.20% and RMS from 90.10% to 95.42% of teacher, although cosine decreases from 0.996582 to 0.994483. The Thorsten whisper improves MAE by 0.56%.

Six recordings have consecutive MAE increases from 2,500 to 3,000 to 3,500, but all remain better than their 2,000 baseline. The largest of these is the flagged Spanish source 13461728374156750135, with increases of 2.05% and 6.04%, totaling approximately 1.066e-4 absolute MAE.

## Silence regression

| Fixed cohort | Step 3,000 passing | Step 3,500 passing |
|---|---:|---:|
| Near-silence | 171/184 | 1/184 |
| Sustained silence after 40 ms | 164/164 | 0/164 |
| Interior near-silence after 800 ms | 50/50 | 0/50 |
| Startup first 20 ms | 0/13 | 0/13 |
| Teacher silence transient at 20 to 40 ms | Not fully passing | 10/10 |

Sustained and interior silence failures are amplitude-only under the existing checks. Sustained residual RMS increases from 1.282 to 3.134 millionths of full scale, while its centered residual changes only from 1.030 to 1.050. Sustained output RMS increases from 8.964 to 12.589 millionths of full scale; teacher RMS is 9.597. Interior residual RMS increases from 1.398 to 3.392, while centered residual changes from 1.235 to 1.269.

These measurements indicate a larger per-window offset contribution. They do not establish a universal signed DC offset or its cause. Threshold excess is about 1.83 to 1.92 millionths of full scale in RMS. The pass-count collapse is real under the fixed thresholds, but does not quantify audibility. No perceptual conclusion is drawn without listening.

The teacher 20 to 40 ms transient improves from 114.13 to 81.54 millionths of full scale in residual RMS and passes all ten checks for the first time. All measured boundary errors improve; final quiet waveform NRMSE on the 12-source boundary panel falls from 0.21755 to 0.18735. This is a new sustained/interior offset regression, while the preceding nonzero-quiet amplitude problem improves. It is not evidence that all quiet behavior or every layer is worsening.

## Integrity and disposition

Checkpoint SHA256: `b5bf6922703a4b1248fe2efceb63065471ebeece717fb548745492ad35b8708e`.

All 15 integrity checks pass, including checkpoint receipt and quality agreement, preserved frozen state, unchanged widths and coefficients, all 90 Adam states at update 3,500 without reset, finite weights and losses, exact source and optimizer-step order, 42,000 total distinct sources and 18,000 new sources, and all 18,000 original-teacher cache comparisons.

The later live snapshot reached update 3,565 with 42,780 total sources and 18,780 passing new teacher comparisons. H100 utilization was 98%. The target producer remained active and had sealed 12,900 of 30,000 extension sources. No failure receipt was present.

TensorBoard API verification exposes all 15 overview runs, quality points at update 3,500 and training progress at update 3,800 (76% of the approved limit). The active waveform chart shows 99.45046%; the all-quiet pass chart shows 43.35692%. Dashboard publication is functioning.

Retain all checkpoints and continue the unchanged recipe to the next scheduled review at 4,000. Specifically check whether the sustained/interior window-offset error persists or worsens; investigate persistent regression before deciding on an intervention. Aggregate correlation alone is not a qualification criterion. No restart, additional pruning, promotion, benchmark or commit is justified by this milestone. The approved run remains bounded to 5,000.
