# Scheduled review at update 3,000

Continue unchanged to the next scheduled review, with amplitude drift explicitly flagged. This is a mixed interval, not sustained overall deterioration and not uniform quality recovery.

Checkpoint SHA256: `cfdff3a2cf9ebecc55fd6b4a8101a9b295494aba5c41ef1ef56f4d5bf7cbbebf`.

Relative to update 2,500, active waveform cosine increased from 0.993581 to 0.994272, mel error decreased 3.31%, and complete-group MSE decreased 9.20%. Waveform MAE increased 0.51%, with 40 recordings improving and 56 worsening. Relative to the preserved update-2,000 control, MAE is still 12.48% lower, mel 10.55% lower, group MSE 22.05% lower, and 83 of 96 recordings improve in MAE.

The principal new concern is excessive amplitude in some recordings, including quiet windows with nonzero teacher audio. All-quiet passing fell from 1123 to 968 of 2544. Amplitude-failure counts increased from 72 to 274; failure categories can overlap. Pooled output RMS on quiet nonzero audio is 106.26% of the teacher. This measures waveform energy, not perceptual loudness.

| Quiet cohort | Residual RMS at 2,500 | At 3,000 | Passing at 2,500 | Passing at 3,000 |
| --- | ---: | ---: | ---: | ---: |
| All quiet | 81.07 | 85.22 | 1123/2544 | 968/2544 |
| Near silence | 5.546 | 5.324 | 171/184 | 171/184 |
| Startup, first 20 ms | 19.93 | 19.29 | 0/13 | 0/13 |
| Teacher transient, 20 to 40 ms | 139.40 | 114.13 | | |
| Source silence after 40 ms | 1.534 | 1.282 | 164/164 | 164/164 |
| Interior near silence | 1.579 | 1.398 | 50/50 | 50/50 |
| Quiet nonzero reference | 83.68 | 88.16 | | |

Residual RMS is in millionths of full scale and cohorts overlap. Only all-quiet and quiet-nonzero residuals worsen. Sustained and interior silence remain within limits with improving residuals. Startup remains unresolved despite a modest decrease in error. The maximum absolute output was 0.98530, below full scale.

The two previously flagged recordings recovered. Whistle 428921 MAE improved 25.42% relative to 2,500 and 0.29% relative to 2,000; cosine reached 0.996582, while RMS remains attenuated at 90.10% of the teacher. The Thorsten whisper's MAE improved 15.93% versus 2,500 and 10.80% versus 2,000; cosine reached 0.986254 and RMS 98.41%.

Several current regressions accompany excessive amplitude despite preserved or improved cosine. Luganda RMS changed from 100.22% to 106.92% of teacher, with MAE increasing 36.36%. Maithili changed from 98.42% to 104.17%, MAE increasing 27.55%; Javanese from 99.38% to 103.84%, MAE increasing 25.27%; Portuguese from 99.06% to 104.00%, MAE increasing 16.90%. Track these recordings alongside the whistle and whisper at the next review.

On the 12-source boundary panel, group and suffix hidden NRMSE measurements improved. Complete stage-4 NRMSE decreased from 0.13128 to 0.12613 and final waveform from 0.11433 to 0.11064. Quiet waveform was nearly flat, 0.217277 to 0.217545, despite improving hidden errors. The quiet-window identity is unchanged.

All 15 independent integrity checks passed. The actual checkpoint SHA and receipt quality agree; frozen-state preservation is recorded; all 90 AdamW states reached 3,000 with unchanged settings; coefficients and widths match the parent; the complete 36,000-source ledger contains no repeats and matches the approved order; all 12,000 new teacher/cache comparisons passed; weights, optimizer state and logged losses are finite. The derived integrity report is on Runpod at `integrity-review-3000.json`.

The trainer, producer supervisor and phase-2 target producer all match their recorded commands. At the live check, training had reached 3,020 and all 12,240 new teacher/cache comparisons passed. Appended-target production had sealed 6,600 sources. Available storage remained sufficient for the outstanding checkpoint and target allocations. No recipe, architecture, source-order or optimizer changes were made.
