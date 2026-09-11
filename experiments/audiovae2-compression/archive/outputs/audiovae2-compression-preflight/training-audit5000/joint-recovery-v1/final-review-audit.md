# Final joint-recovery review

The authorized 1,000-update joint run completed at step 5625. The conditional joint-versus-freeze experiment is not warranted under the agreed trigger: none of the 29 material metric/source flags at 750 updates repeats at the final review, and the progress gate does not report a stall. Six new flags remain across five sources. Keep the final checkpoint and the earlier review checkpoints; stop at the authorized budget without automatically freezing or promoting a model.

| Measure | Start4625 | Final5625 | Change |
|---|---:|---:|---:|
| Active waveform correlation |0.976007|0.979521|+0.003514|
| Waveform MAE |0.00423436|0.00396681|−6.32%|
| Mel error |0.335225|0.307806|−8.18%|
| Group-output MSE |0.0116247|0.00977642|−15.90%|
| Quiet residual RMS |0.000176647|0.000161971|−8.31%|
| Near-silence residual RMS |0.0000291007|0.0000261471|−10.15%|
| Quiet windows passing |138/2544|220/2544|+82|
| Near-silence windows passing |18/184|47/184|+29|
| Full-scale overshoot samples |0|0|Unchanged|

The final waveform MAE improves on both the 500-update and 750-update checkpoints. The pooled waveform MSE remains about 2% higher than at 500 updates, so the final result is not best in every statistic. Compared with start, MAE improves in 92/96 sources, mel in 95/96 and complete group MSE in all96. The target correlation of 0.99 has not been reached, and most quiet windows still fail the engineering checks. These results demonstrate progress, not teacher parity or perceptual qualification.

## What happened to the 750-update regressions

The earlier Telugu, Luganda, Polish and Azerbaijani attenuation recovers. Their final student/teacher active RMS ratios are approximately1.018,1.013,1.031 and1.031. The amplified `freesound:402835` source recovers from1.107 to1.003. Their earlier metric/source flags no longer satisfy the locked material-regression comparisons. This is direct evidence against using that single mixed review to freeze stages2 and3.

The final six flags consist of three active-level flags and three MAE flags. The source labels below come from the saved pilot manifest; all three expressive sources score their entire prepared recording, and their labels are source-level annotations rather than timed event labels.

| Source | Label | Active RMS ratio | MAE versus start |
|---|---|---:|---:|
|freesound:236508|Whispering|0.905|−8.76%|
|freesound:343940|Laughter|0.869|+1.59%|
|freesound:343960|Yell|0.889|−6.72%|

The laughter source also has a new MAE flag versus the preceding review. The other new MAE flags are a Thorsten whispered source and the Dogri source. These require continued attention in any later approved work; better aggregate reconstruction does not remove the quieter expressive-output issue. No quiet-RMS source flag remains at the final review.

## Boundaries and integrity

The fixed12-source panel retains exact stage1 equality. From750 to1,000 updates, complete stage4 quiet NRMSE improves0.22041→0.21123 and active NRMSE0.35577→0.34927. Active waveform NRMSE also improves0.21487→0.20377. Quiet waveform NRMSE on this smaller panel worsens slightly0.43047→0.43701 despite better hidden quiet errors; this does not contradict the improved96-source pooled quiet RMS because the source subset and normalization differ. It remains a reminder that hidden MSE does not uniquely determine waveform quality.

The run consumed12,000 distinct additional sources, and its12,000 teacher/cache checks passed. The saved receipts confirm preserved teacher, frozen student modules and original files, with no failure. Optimizer and RNG state were checkpointed. This review used saved reports only; it launched no model work and changed no training source.
