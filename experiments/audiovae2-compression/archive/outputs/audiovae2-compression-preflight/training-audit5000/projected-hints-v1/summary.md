# Projected feature hints: result and decision

The bounded comparison completed successfully. The two intermediate teacher hints improved feature matching but did not improve final reconstruction enough to adopt. Preserve the previous candidate and keep the main run paused. Both experimental checkpoints remain available separately.

Both arms started from the retained accumulation-12 checkpoint at optimizer step 4,625 and finished at 4,750. Each used the same 1,500 previously used diagnostic recordings once, with 125 updates and unchanged student optimizer settings. This was a controlled debugging comparison, not fresh-data training. Total execution time, including calibration and evaluation, was 252 seconds on the Runpod GPU.

## Fixed 96-recording development panel

| Metric | Starting candidate | Existing recipe after 125 updates | With two hints after 125 updates |
|---|---:|---:|---:|
| Active waveform correlation, higher is better | 97.6007% | 97.6601% | 97.6598% |
| Waveform MAE, lower is better | 0.00423436 | 0.00417039 | 0.00417764 |
| Mel error, lower is better | 0.335225 | 0.329310 | 0.330991 |
| Complete stage-4 output MSE, lower is better | 0.0116247 | 0.0113457 | 0.0108878 |
| Quiet residual RMS, lower is better | 0.000176647 | 0.000174378 | 0.000173869 |
| Quiet windows passing | 138 / 2,544 | 165 / 2,544 | 150 / 2,544 |
| Near-silence windows passing | 18 / 184 | 3 / 184 | 1 / 184 |
| Samples exceeding full scale | 0 | 0 | 0 |

Compared with the existing recipe at equal exposure, hints reduced complete-block MSE by 4.04%, but waveform MAE increased 0.174% and mel error increased 0.511%. These are small differences, not evidence of a dramatic audible degradation. They provide no output-quality benefit that would justify adoption. The 0.99 correlation goal remains unmet.

The slightly lower pooled quiet RMS does not mean silence was fixed: fewer windows met the same checks. Both continuations also lost near-silence passes compared with their common starting candidate. That regression matters even though ordinary waveform and mel averages improved with further training.

## What changed inside the student

With the initial projection maps held fixed for evaluation, stage-3-up feature MSE fell 9.63% and stage-4-up feature MSE fell 13.92% relative to the baseline endpoint. Thus the student representation changed in the intended direction; the apparent improvement was not solely the alignment maps adapting.

That representation improvement did not translate into improved waveform fidelity through the frozen suffix. The result demonstrates a limitation of these two feature objectives at this strength and update budget. It does not prove that every projected-distillation method fails, that the compressed model lacks sufficient capacity, or that a longer run would necessarily fail.

Per-recording results support the aggregate decision:

- Complete-block MSE improved on 85 of 96 recordings, while waveform MAE worsened on 55 and mel error worsened on 74. The machine-readable audit distinguishes differences within the existing numerical reproduction tolerance.
- Quiet RMS worsened on 39 of the 68 recordings containing quiet windows, despite the small pooled improvement.
- Active amplitude became closer to the teacher on 61 of 94 eligible recordings. This is a benefit, but the three speech diagnostic trajectories became more variable, and the whistling trajectory worsened.
- For the whistling case, final active RMS was 86.94% of the teacher with the existing recipe and 86.43% with hints. Its mean waveform MAE across all 26 predefined observations worsened about 3%.
- The 11 recordings identified as expressive material also had slightly worse pooled waveform MAE with hints. These are recording labels, not measured counts of expressive event duration.

## Validation and retained artifacts

All 12 focused tests passed locally and on Runpod. Disabling the hints reproduced the original training update, parameters and AdamW moments exactly in the focused control. Both real arms restored identical initial weights, moments and RNG; their initial 96-recording reports were exactly equal. Both consumed 176,445,657 scored training samples in identical source order. All 3,000 teacher/cache checks were bitwise equal. The teacher, frozen decoder stages and original files were preserved.

The projections are separate training modules. They never feed into the decoder or frozen suffix, and no decoder state keys or inference layers were added. No new RTF benchmark or production export was performed. Neither endpoint was promoted, and no additional pruning or training run was started.

The subsequent approved native-upsampler investigation has now completed. It establishes strong downstream coadaptation but does not yield an acceptable replacement; see [results and recommendation](../native-up4-reconstruction-v1/summary.md). Neither investigation gives a reason to make another channel cut before recovery is demonstrated.

Details: [protocol](protocol.md), [independent result audit](independent-analysis.md), [execution audit](execution-audit.md), [raw comparison](results/completed.json), [per-source analysis](results/independent-analysis.json).
