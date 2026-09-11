# First joint-recovery review

The scheduled review at step4875 supports continuing the authorized joint run. It covers250 additional updates and3,000 distinct training sources from the restored4625 checkpoint. These are deterministic comparisons on the same96 development sources, not perceptual quality or CPU speed claims.

| Measure | Start4625 | Review4875 | Change |
|---|---:|---:|---:|
| Active waveform correlation |0.9760071|0.9775265|+0.0015193|
| Waveform MAE |0.00423436|0.00409770|−3.23%|
| Mel error |0.335225|0.326517|−2.60%|
| Complete group-output MSE |0.0116247|0.0109427|−5.87%|
| Quiet residual RMS |0.000176647|0.000171836|−2.72%|
| Quiet windows passing |138/2544|156/2544|+18|
| Near-silence residual RMS |0.0000291007|0.0000292767|+0.605%|
| Near-silence windows passing |18/184|5/184|−13|
| Active student/teacher RMS ratio |0.968584|0.976321|Closer to1|
| Full-scale overshoot samples |0|0|Unchanged|

The near-silence pass-count decline remains an explicit alert. Its pooled residual increase is only1.76e-7, across175,875 exactly matched near-silence samples. Near-silence RMS worsens in13 of15 source subsets, but the largest source increase is1.14e-6. The direction deserves monitoring; the count change alone does not establish a large new noise burst or an audible regression. The frozen engineering gates require both10% relative and1e-5 absolute residual worsening before a material-regression flag.

## Source checks

Waveform MAE improves in92/96 sources, mel in93/96, and group MSE in95/96. The largest MAE deterioration is the Kannada source `kn_in:train:14687240407945660913.wav`: +1.19%, or3.74e-5 absolute. The three other MAE deteriorations are smaller. Quiet RMS improves in56 of68 quiet-bearing sources. The largest quiet-RMS rise is7.70e-6 on only126 valid samples; among sources with at least20ms of quiet audio, the largest rise is4.76e-6 on the Dogri source. None reaches the material source guard.

Active amplitude error, measured as absolute dB distance from teacher RMS on teacher-active samples, improves in76/94 active sources. Its worst worsening is0.193dB, below the0.5dB review threshold. Thus this review does not show the broad gain drift seen in the earlier unstable continuation.

## Boundary checks and limits

The12-source boundary panel preserves exactly equal stage1 outputs in all, quiet and active regions. At the full stage4 output, quiet cosine is0.97284; after the frozen suffix, quiet waveform cosine is0.89097. This localizes remaining functional error to the modified box and its downstream effect, but these numbers alone do not establish a new change in suffix sensitivity. There is no new step4625 boundary report from this run, so these internal values are not presented as a paired trend.

Stage2 andstage3 compare only retained teacher coordinates to narrower, adapted student states. Their relatively low cosine scores are not whole-representation fidelity scores and do not justify freezing those stages. The complete stage4 output and final waveform remain the meaningful shared-boundary measurements.

The checkpoint receipt confirms3,000 teacher/cache checks, preserved frozen state, and saved optimizer/RNG state. The restored initial96-source quality check passed. The gate returns `continue`, with a near-silence count alert and no material-regression flags. Continuing to the next scheduled review is consistent with the approved protocol; no extra training variant, layer change, or model check was launched for this audit.
