# Final step 5,000 review

The authorized same-width recovery is finished. Retain its checkpoint and proceed with the approved identical-teacher control. No further pruning cut or recovery training is recommended before that control is interpreted.

These are compact calculations from the saved Runpod reports, not a new model evaluation. Raw reports remain remote. All comparisons use the same 96 development recordings and identical quiet-window identities.

| Metric | 2,000 | 4,500 | 5,000 |
|---|---:|---:|---:|
| Mean active waveform cosine | 0.991865 | 0.995225 | 0.995233 |
| Waveform MAE | 0.00265208 | 0.00200070 | 0.00196372 |
| Mel error | 0.160883 | 0.129264 | 0.126230 |
| Full group-output MSE | 0.00240390 | 0.00152196 | 0.00146269 |
| Quiet residual RMS, millionths of full scale | 87.866 | 71.412 | 70.263 |
| Active output RMS relative to teacher | 99.947% | 97.676% | 98.611% |

The final 500 updates improve MAE 1.85%, mel error 2.35%, group MSE 3.89% and aggregate quiet residual RMS 1.61%. MAE improves on 69/96 recordings, mel on 84/96, and group MSE on 80/96. Every recording remains better than at 2,000 on all three errors; pooled improvements versus 2,000 are 25.96%, 21.54% and 39.15%, respectively. Mean cosine barely changes in the last interval. It is an active-waveform similarity measure, not perceptual accuracy.

## Quiet audio

| Cohort | Passing at 4,500 | Passing at 5,000 | Residual RMS at 4,500 → 5,000, millionths |
|---|---:|---:|---:|
| All quiet | 1380/2544 | 1511/2544 | 71.412 → 70.263 |
| Near-silence | 170/184 | 171/184 | 5.667 → 6.935 |
| Startup first 20 ms | 0/13 | 0/13 | 20.805 → 22.004 |
| Source-silence teacher transient, 20 to 40 ms | 10/10 | 10/10 | 85.264 → 75.175 |
| Sustained source silence after 40 ms | 164/164 | 164/164 | 1.085 → 3.773 |
| Interior near-silence after 800 ms | 49/50 | 50/50 | 1.346 → 3.843 |
| Quiet with nonzero original source | 1206/2359 | 1337/2359 | 73.930 → 72.772 |

The passing counts conceal a remaining difference. Sustained and interior residuals worsen while remaining below the absolute 10-millionths residual limit. Sustained output RMS is 6.103 millionths versus teacher 9.597; interior output is 6.152 versus teacher 9.635. Their centered residuals barely change, from 1.043 to 1.057 and 1.227 to 1.272 millionths. The increased per-window mean-error component accompanies lower output level. Passing these provisional thresholds therefore does not establish exact teacher reconstruction. The amplitude condition is an upper bound, not a two-sided demand for equal RMS.

The previous Spanish interior amplitude failure recovers. The 13 remaining near-silence failures are the startup windows, failing both checks. Their residual RMS worsens 5.76% in this interval, although output-limit excess decreases from 12.884 to 12.270 millionths. The teacher's 20 to 40 ms transient improves and remains passing, with output RMS 574.029 versus teacher 632.831 millionths.

Across all quiet windows, 1,000 failures are residual-only, 31 fail both checks, and two are amplitude-only. Forty of 68 recordings containing quiet windows improve residual RMS. Thirteen have consecutive quiet-RMS increases across 4,000, 4,500 and 5,000. Punjabi `pa_in:train:16693733152628588519.wav` retains the previously flagged trend: 94.503 → 97.223 → 102.907 millionths, with quiet failures 11 → 11 → 14. Its active MAE and cosine nevertheless improve. Other retained quiet regressions include Bodo `bodo:3659174697289264_chunk_3.flac` (120.421 → 120.745 → 143.870) and Somali `so_so:train:14842603451866844632.wav` (94.091 → 98.390 → 112.034). The cohorts overlap and are not a partition; pass counts are not audibility ratings.

## Recordings and boundary evidence

Whistle `freesound:428921` reverses its quiet regression: residual RMS falls 60.645 → 53.647 millionths. Its 17 quiet failures remain, while MAE improves 0.00033461 → 0.00031507, cosine reaches 0.996058, and active RMS ratio improves 96.432% → 99.711%. Peak is 0.065611.

Thorsten whisper `thorsten_emotional:whisper/37773d61d5c91c12d49e55e2b041c0cd.wav` is mixed: MAE worsens 0.94% to 0.00660063 and cosine falls slightly to 0.987807, while quiet residual RMS improves 137.877 → 130.740 millionths and active RMS ratio improves 96.553% → 97.009%. Its eight quiet failures remain. Peak is 0.982558.

`freesound:402835` is the only recording with consecutive whole-recording MAE increases across the last two intervals: +0.52%, then +10.86%. It remains 13.70% better than at 2,000. Other expressive regressions deserve preservation in the final report: `freesound:92375` MAE rises 9.30% and cosine falls 0.993020 → 0.988635; `freesound:343960` MAE rises 8.73% and cosine falls 0.967587 → 0.957373, despite improved mel error. These prevent a claim of uniform final-checkpoint superiority. Global maximum absolute output is 0.992325, with zero overshoot samples.

On the fixed twelve-recording boundary panel, stage 1 remains exact. Full stage-4 NRMSE improves 0.113675 → 0.113189 overall and 0.073992 → 0.073113 on quiet samples. Waveform NRMSE improves 0.110054 → 0.102205 overall and 0.190875 → 0.188005 on quiet samples. Some intermediate quiet errors worsen: stage 5 moves 0.083732 → 0.087953 and stage 6 moves 0.084477 → 0.089585. Internal stage-2/3 values compare selected coordinates only; their interpretation differs from full external boundaries. These measurements do not uniquely identify the cause of startup error or prove a capacity limit.

RMS ratios use the existing `overview_window_metrics.by_source` energy fields: `sqrt(sum(active_student_energy)/sum(active_teacher_energy))`, pooled or restricted to a named recording. No subjective loudness or new CPU performance claim is made. The identical-teacher control remains necessary to separate metric behavior, original native execution, cached targets and any training-path sensitivity before deciding what to change next.
