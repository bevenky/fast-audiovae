# Scheduled review at update 2,500

Continue the approved recovery toward 5,000 with the same recipe. This milestone shows broad improvement, with specific whistle and whisper regressions to track. It does not establish teacher parity or authorize another pruning cut.

Checkpoint SHA256: `bf89d89272b401971e26301e6070081a4b63919d8e80ed0b93c0914c10a30a9e`.

Compared with the preserved step-2,000 checkpoint, active waveform cosine increased from 0.991865 to 0.993581. Waveform MAE fell 12.92%, mel error 7.48%, complete-group MSE 14.15%, and all-quiet residual RMS 7.73%. Waveform MAE improved on 92 of the same 96 recordings; mel improved on all 96. Active pooled RMS relative to the teacher changed from 99.95% to 99.37%.

| Quiet cohort | Residual RMS at 2,000 | At 2,500 | Passing at 2,000 | Passing at 2,500 |
| --- | ---: | ---: | ---: | ---: |
| All quiet | 87.87 | 81.07 | 1066/2544 | 1123/2544 |
| Near silence | 5.499 | 5.546 | 171/184 | 171/184 |
| Startup, first 20 ms | 19.76 | 19.93 | 0/13 | 0/13 |
| Teacher transient, 20 to 40 ms | 158.80 | 139.40 | | |
| Source silence after 40 ms | 1.407 | 1.534 | 164/164 | 164/164 |
| Interior near silence | 1.516 | 1.579 | 50/50 | 50/50 |
| Quiet nonzero reference | 90.64 | 83.68 | | |

Residual RMS is in millionths of full scale. These cohorts overlap. Small sustained/interior residual increases are below passing limits, with zero output-limit excess and decreasing centered residual. They are not evidence of renewed excess noise. Startup remains unresolved.

Whistle 428921 worsened in waveform MAE by 33.70%; its RMS relative to teacher fell from 92.25% to 88.39%, and cosine fell from 0.994777 to 0.991189 even though mel improved 4.07%. The Thorsten whisper's MAE worsened 6.10% and cosine fell from 0.983239 to 0.981244, while RMS moved toward the teacher from 94.96% to 97.44%. The other two MAE regressions were very small. Keep per-recording shape and amplitude checks separate.

On the 12-source boundary panel, every measured all-audio and quiet boundary NRMSE improved. Complete stage-4 NRMSE fell from 0.14361 to 0.13128, final waveform from 0.13831 to 0.11433, and quiet waveform from 0.22697 to 0.21728. Stage 1 remained exactly identical.

Independent checkpoint checks passed: actual SHA matches the receipt; receipt quality matches the report; frozen-state preservation is recorded; all 90 AdamW states reached 2,500 without resetting; settings, coefficients and widths match the parent; checkpoint history contains the exact original 24,000 sources followed by 6,000 new sources; all 6,000 teacher/cache comparisons passed; and weights, optimizer state and logged losses were finite. The derived integrity report remains on Runpod at `integrity-review-2500.json`.

The first target-production phase completed its 3,000 missing original targets. The second producer is active, preparing the separate appended source cache. Trainer and both producer process identities were verified. No architecture, recipe, data order or optimizer change was made in this review.
