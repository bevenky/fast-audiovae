# Projected hints: independent saved-result audit

**Do not adopt these two hint losses as a quality improvement from this experiment.** They make intermediate teacher matching better, including under the original frozen readouts, but the common waveform and spectral endpoint does not improve. Quiet pass counts and the prespecified whistle trajectory worsen. Both experimental checkpoints remain useful evidence; neither was automatically promoted.

The comparison uses the retained accumulation12 checkpoint at optimizer step4625. Both arms consume the same1,500 ordered diagnostic sources with125 updates, ending at4750. Source reuse across diagnostic arms is explicit. These are development-panel results, not an untouched final test.

Integrity: comparison_valid=True; initial reports exactly equal=True; teacher/window coverage invariant=True. Each arm has 176,445,657 scored training samples. All teacher/cache targets bitwise equal: baseline=True, hints=True.

## Common96-source endpoint

| Metric | Initial | Baseline | Two hints |
|---|---:|---:|---:|
| mae | 0.00423436186 | 0.00417039103 | 0.00417764102 |
| mse | 0.000226950244 | 0.000220800562 | 0.000222137886 |
| nonquiet_cosine_mean | 0.976007129 | 0.97660131 | 0.976598015 |
| mel | 0.335224718 | 0.329309583 | 0.330991149 |
| mel_linear | 0.00311477832 | 0.00303007895 | 0.00303056906 |
| mel_log | 0.332109928 | 0.326279491 | 0.327960581 |
| group_mse | 0.0116247346 | 0.0113457317 | 0.0108877887 |
| original_weighted_objective | 0.00456624872 | 0.00449573425 | 0.00449984579 |
| quiet_residual_rms_mean | 0.000176646907 | 0.00017437838 | 0.000173869493 |
| quiet_passing_windows | 138 | 165 | 150 |
| near_silence_passing_windows | 18 | 3 | 1 |
| peak_abs_max | 0.970560253 | 0.973435044 | 0.971601903 |
| overshoot_samples | 0 | 0 | 0 |

MAE change versus baseline: 0.1738%. Original weighted-objective change: 0.0915%. Quiet support=2544; near-silence support=184.

## Paired sources

Counts retain every measured direction; the JSON separately marks differences within existing numerical reproduction tolerance. That tolerance is not a perceptual acceptance criterion.

| Metric | Better | Worse | Equal |
|---|---:|---:|---:|
| mae | 41 | 55 | 0 |
| mse | 48 | 48 | 0 |
| cosine | 48 | 46 | 0 |
| mel | 22 | 74 | 0 |
| mel_linear | 43 | 53 | 0 |
| mel_log | 22 | 74 | 0 |
| group_mse | 85 | 11 | 0 |
| quiet_rms | 29 | 39 | 0 |
| quiet_passes | 3 | 12 | 81 |
| near_silence_passes | 0 | 1 | 95 |

## Full-panel active amplitude

| Metric | Initial | Baseline | Two hints |
|---|---:|---:|---:|
| eligible_sources | 94 | 94 | 94 |
| pooled_rms_gain | 0.968584416 | 0.971708899 | 0.971566425 |
| median_gain | 0.97175428 | 0.984622743 | 0.991657929 |
| below_teacher | 90 | 79 | 65 |
| above_teacher | 4 | 15 | 29 |
| mean_absolute_log_gain_error | 0.0347974869 | 0.0266376333 | 0.0238634858 |

## All26 matched trajectory points

The panel contains three speech recordings and one whistling recording. It does not represent four expressive categories. Every predefined snapshot is included, including the common starting point.

| Cohort | Measure | Baseline | Two hints | Change |
|---|---|---:|---:|---:|
| three_speech_recordings | mean_absolute_log_gain_error | 0.0110609986 | 0.0119282723 | 7.8408% |
| three_speech_recordings | gain_population_std | 0.0102279183 | 0.0115239388 | 12.6714% |
| three_speech_recordings | mean_mae | 0.00379181923 | 0.00380946667 | 0.4654% |
| one_whistle_recording | mean_absolute_log_gain_error | 0.137135923 | 0.148841665 | 8.5359% |
| one_whistle_recording | gain_population_std | 0.0579318186 | 0.0581822462 | 0.4323% |
| one_whistle_recording | mean_mae | 0.000780964483 | 0.000804444479 | 3.0065% |

## Readout interpretation

| Arm | Hint | Initial | Final learned readout | Final original frozen readout |
|---|---|---:|---:|---:|
| baseline | stage3_up | 0.00410556515 | 0.00410805799 | 0.00410805799 |
| baseline | stage4_up | 0.00173989757 | 0.00174102171 | 0.00174102171 |
| two_hints | stage3_up | 0.00410556515 | 0.00370662615 | 0.00371228742 |
| two_hints | stage4_up | 0.00173989757 | 0.00150745371 | 0.00149873369 |

A learned-readout decrease can include changes in both the student representation and the auxiliary map. The frozen-initial-readout score holds the map fixed, but still does not establish downstream waveform quality. Common waveform, spectral, amplitude and quiet metrics remain the decision criteria. The readouts are external to the decoder and add no deployed inference operations.

## What the paired result establishes

- Internal progress is not merely the trainable projectors absorbing error. With both original readouts frozen for scoring, stage3 hint MSE improves9.58% and stage4 improves13.86% versus the initial checkpoint. Fixed-readout loss improves on96/96 and94/96 sources respectively. The directly measured full group-boundary MSE also improves4.04% versus baseline, on85/96 sources.
- That improvement does not carry through to final reconstruction: waveform MAE is worse on55/96 sources and common mel error on74/96. The mean active correlation is effectively unchanged, with48 sources improving and46 worsening. The original weighted objective also worsens slightly. The run therefore supplies evidence against using these particular hidden-state MSE hints as a proxy for waveform quality.
- Quiet residual RMS improves only0.292% when pooled; it worsens on39/68 sources with quiet support. Spanish quiet RMS rises12.90%, Freesound236508 rises10.74%, and Welsh rises8.11%. Twelve recordings lose quiet passing windows while three gain some. The two lost near-silence passes are both in Freesound277554. Both arms lose near-silence passes compared with the starting checkpoint:18 initially,3 baseline,1 hints. Thresholds and window coverage are unchanged.
- Full-panel amplitude distance improves on61/94 active sources and worsens on33. This is a limited favorable result. It does not contradict the worse three-speech-recording temporal trajectory, which measures a different subset over all26 points. The whistle finishes at86.43% of teacher RMS versus86.94% for baseline, with1.58% worse endpoint MAE; its trajectory mean MAE worsens3.01%. All61 of its quiet windows still fail in both arms.
- Across ten Freesound recordings plus the named Thorsten whisper recording, sample-pooled MAE changes from0.0171896641 to0.0172580574; mean active correlation changes from0.935691223 to0.93538415. These are recording-level labels, not measurements of event-time occupancy.
- No full-scale overshoots occur in either arm. Maximum sample level is reported descriptively; a smaller peak is not automatically better fidelity.

This is one125-update, paired-source diagnostic, not a proof that all feature distillation fails or that the narrowed decoder has reached a capacity floor. It supports keeping the present hint recipe out of the accepted training path. It does not justify a speculative coefficient change or another architecture change without a separate proposal.

The machine-readable audit includes every paired source, cohort, preserved input hash and integrity check. No model inference or training was executed by this analysis.
