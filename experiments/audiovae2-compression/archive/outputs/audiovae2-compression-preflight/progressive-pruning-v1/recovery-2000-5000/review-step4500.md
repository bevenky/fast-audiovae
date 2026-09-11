# Step 4,500 review

Continue the unchanged run to the authorized 5,000-update endpoint, then run the queued identical-teacher control. The latest review shows broad waveform recovery with some smaller, separate quiet-audio regressions. No new pruning cut, objective change or additional training is proposed here.

This review uses compact calculations performed on the saved Runpod reports. Full reports and logs remain remote. The fixed 96 recordings and quiet-window identity are unchanged.

| Metric | 2,000 | 4,000 | 4,500 |
|---|---:|---:|---:|
| Mean active waveform cosine | 0.991865 | 0.994044 | 0.995225 |
| Waveform MAE | 0.00265208 | 0.00222129 | 0.00200070 |
| Mel error | 0.160883 | 0.132218 | 0.129264 |
| Full group-output MSE | 0.00240390 | 0.00160229 | 0.00152196 |
| Quiet residual RMS, millionths of full scale | 87.866 | 70.758 | 71.412 |
| Active output RMS relative to teacher | 99.947% | 97.472% | 97.676% |

Versus 4,000, MAE improves 9.93%, mel error 2.23%, and group MSE 5.01%. MAE improves on 91/96 recordings, mel on 81/96, and group MSE on 70/96. All 96 recordings remain better than at 2,000 on each of these three errors. No recording has consecutive MAE increases across 3,500 to 4,000 to 4,500. Cosine is an active-waveform similarity measure, not perceptual accuracy.

## Quiet audio

The seven cohorts overlap. Residual RMS below is sample pooled; passing counts count windows and should not be added across cohorts.

| Cohort | Passing at 4,000 | Passing at 4,500 | Residual RMS at 4,000 → 4,500, millionths |
|---|---:|---:|---:|
| All quiet | 1367/2544 | 1380/2544 | 70.758 → 71.412 |
| Near-silence | 171/184 | 170/184 | 5.694 → 5.667 |
| Startup first 20 ms | 0/13 | 0/13 | 20.612 → 20.805 |
| Source-silence teacher transient, 20 to 40 ms | 0/10 | 10/10 | 92.699 → 85.264 |
| Sustained source silence after 40 ms | 164/164 | 164/164 | 1.402 → 1.085 |
| Interior near-silence after 800 ms | 50/50 | 49/50 | 1.523 → 1.346 |
| Quiet with nonzero original source | 1203/2359 | 1206/2359 | 73.211 → 73.930 |

Sustained silence continues to improve. Its output RMS is 9.891 millionths versus teacher 9.597, with no output-limit excess. Interior output RMS is 9.953 versus teacher 9.635. Its one amplitude-only failure is in `es_419:train:13461728374156750135.wav`; the cohort's pooled RMS excess above its per-window output limits is only 0.028 millionths. The remaining 13 near-silence failures are the existing startup windows, failing both residual and amplitude checks. Startup output RMS rises from 22.475 to 23.525 millionths versus teacher 9.334. This remains unresolved.

The small aggregate quiet-RMS increase does not mean every recording worsens, but only 28/68 recordings containing quiet windows improve that continuous metric. The largest positive quiet-error-energy change is `3242-67168-0001`: residual RMS 108.444 → 119.767 millionths and failures 8 → 18. Two sources have consecutive quiet-only RMS increases since 3,500: whistle `freesound:428921` (54.218 → 58.509 → 60.645 millionths) and Punjabi `pa_in:train:16693733152628588519.wav` (87.884 → 94.503 → 97.223). These are distinct from whole-recording MAE and do not establish audibility under these provisional engineering limits.

## Recordings and boundaries

Whistle `freesound:428921` improves MAE from 0.00042276 to 0.00033461 and cosine from 0.994416 to 0.995996. Its active RMS ratio improves from 94.539% to 96.432%, despite the quiet-window regression noted above; quiet failures increase 15 → 17. Its peak is 0.061702.

Thorsten whisper `thorsten_emotional:whisper/37773d61d5c91c12d49e55e2b041c0cd.wav` improves MAE 0.00702601 → 0.00653896 and cosine 0.985102 → 0.988219. Active RMS ratio improves 96.134% → 96.553%; its eight quiet failures remain. Peak is 0.983434. Previously watched Luganda, Maithili, Javanese and Portuguese recordings all improve MAE and move their active RMS ratios closer to the teacher in this interval.

The largest relative MAE regression is the very low-amplitude Tajik recording `tg_tj:train:11851591230613317287.wav`: +13.05%, but only 0.00000853 absolute MAE; active RMS ratio rises 101.741% → 104.429% while cosine improves. The other four MAE increases are below 0.69%. Global peak is 0.990894 on `freesound:92375`, with zero overshoot samples.

On the fixed twelve-recording boundary panel, stage 1 stays exact. Full stage-4 NRMSE improves 0.118360 → 0.113675 overall and 0.075948 → 0.073992 on quiet samples. Waveform NRMSE improves 0.122286 → 0.110054 overall but worsens 0.187300 → 0.190875 on quiet samples. Stage-2/3 selected-coordinate errors also improve; these are partial-coordinate diagnostics, not full internal-representation fidelity. Better group error therefore still does not guarantee every downstream quiet metric improves.

Active RMS ratios above are calculated from `overview_window_metrics.by_source`: `sqrt(sum(active_student_energy)/sum(active_teacher_energy))`, pooled across recordings or restricted to the named recording. They describe measured waveform energy, not subjective loudness.

The parent independently verified all 15 integrity checks: checkpoint SHA-256 `c78ec9da4580fc181994b5807466529ac8ad112dfd6888ff2f42fdfe46983294`, 54,000 distinct ordered sources, all 30,000 new teacher/cache checks within the existing tolerance, and all 90 finite Adam states at step 4,500 with unchanged settings. That tolerance is not proof of bitwise low-level target equality. The post-5,000 control will examine native identity, cached targets and training-path sensitivity separately.
