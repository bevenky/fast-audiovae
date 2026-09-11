# Native upsampler reconstruction: independent audit

**Retain the existing trained checkpoint. None of the fitted replacements improves its audio quality.** This experiment does establish a concrete mechanism: the adapted stage4 residual stack expects its jointly learned upstream representation. Substituting an original-teacher representation into it can produce a large gain error even when local features or waveform correlation look better.

All calculations below use saved JSON reports only. Native fitting uses72 calibration sources and a fixed shared-bias causal stride2/kernel4 operator;96 development sources are evaluated without fitting on them. Original initialization and the retained trained4625 checkpoint are separate bases. Teacher injections are diagnostic controls, not deployable replacements.

Integrity: 1224 original saved teacher/cache checks plus72 receipt-recovery checks; all bitwise=True; all source/window support invariant; both native-fold checks pass. No neural training, original checkpoint mutation or automatic promotion occurred. The teacher-retained-input solver receipt was originally overwritten by its calibration score, then recovered in a separate identical72-source fit without new development scoring; both original score and launch hashes remain verified.

## Final waveform on the common96 sources

| Variant | MAE | MSE | Active cosine | Mel | Group MSE | Quiet passes | Near passes | Peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sliced_initial-baseline | 0.0248091392 | 0.003511611 | 0.327810212 | 1.67465913 | 0.0960261868 | 0/2544 | 0/184 | 0.00930834 |
| sliced_initial-native_fit | 0.0169902371 | 0.00205369536 | 0.783736918 | 0.82184422 | 0.0154981252 | 0/2544 | 0/184 | 0.3314736 |
| sliced_initial-teacher_up4 | 4.80377872e-09 | 9.525298e-17 | 1 | 2.93637618e-06 | 1.38087833e-15 | 2544/2544 | 184/184 | 0.9940209 |
| trained4625-baseline | 0.00423436186 | 0.000226950244 | 0.976007129 | 0.335224718 | 0.0116247346 | 138/2544 | 18/184 | 0.9705603 |
| trained4625-native_fit | 0.0203497533 | 0.00191983094 | 0.962998806 | 0.507895648 | 0.0152262954 | 0/2544 | 0/184 | 0.976414 |
| trained4625-teacher_up4 | 0.0197829831 | 0.00171968893 | 0.993828011 | 0.363530815 | 0.00919677718 | 53/2544 | 1/184 | 0.9958826 |
| trained4625-native_fit_original_residuals | 0.00490317641 | 0.000273645883 | 0.970514858 | 0.459560782 | 0.0070759964 | 89/2544 | 5/184 | 0.9182333 |
| teacher-group-end-control | 0 | 0 | 1 | 0 | 0 | 2544/2544 | 184/184 | 0.9940209 |

Peaks are descriptive, not directional quality scores. Quiet, near-silence and active labels use the unchanged teacher masks.

## Local native-up4 output reconstruction

| Basis | Region | Calibration MSE | Development MSE | Development change vs native baseline |
|---|---|---:|---:|---:|
| sliced_initial | all | 0.00364264504 | 0.00379348072 | -86.9267% |
| sliced_initial | active | 0.00413390957 | 0.00422915486 | -85.0987% |
| sliced_initial | quiet | 0.00193447083 | 0.00221850988 | -92.9156% |
| sliced_initial | near_silence | 0.00287239208 | 0.00281729829 | -97.3244% |
| sliced_initial | startup_40ms | 0.00475486028 | 0.00479373132 | -89.3378% |
| trained4625 | all | 0.00141545268 | 0.00149656185 | -81.9114% |
| trained4625 | active | 0.00165576602 | 0.00172527046 | -80.4823% |
| trained4625 | quiet | 0.000579860008 | 0.000669775578 | -89.2447% |
| trained4625 | near_silence | 0.00104973407 | 0.000940377299 | -94.9140% |
| trained4625 | startup_40ms | 0.00369405321 | 0.0044832526 | -60.5544% |

The JSON separately reports phase0 and phase1, source-level SSE changes, weighted support, and teacher energy. A reduced local feature error is not itself an audio-quality gain.

## Teacher retained-input control

| Region | Fit MSE | Held-out MSE | Held-out SSE / teacher energy |
|---|---:|---:|---:|
| all | 0.000287536407 | 0.000304314714 | 0.00830982048 |
| active | 0.00032731422 | 0.00034447025 | 0.00898056055 |
| quiet | 0.000149225109 | 0.000159151642 | 0.0052448103 |
| near_silence | 0.000100647021 | 9.47479761e-05 | 0.000972477859 |
| startup_40ms | 0.000501244091 | 0.000493467281 | 0.0108768239 |

This last control uses original teacher retained coordinates. Its fit is never installed into the adapted model; it measures local predictability without accumulated student drift.

## Paired source directions

| Comparison | Metric | Better | Worse | Equal |
|---|---|---:|---:|---:|
| sliced_initial-native_fit-versus-baseline | mae | 85 | 11 | 0 |
| sliced_initial-native_fit-versus-baseline | mel | 96 | 0 | 0 |
| sliced_initial-native_fit-versus-baseline | cosine | 94 | 0 | 0 |
| sliced_initial-native_fit-versus-baseline | group_mse | 96 | 0 | 0 |
| sliced_initial-native_fit-versus-baseline | quiet_rms | 14 | 54 | 0 |
| sliced_initial-native_fit-versus-baseline | quiet_passes | 0 | 0 | 96 |
| sliced_initial-native_fit-versus-baseline | near_passes | 0 | 0 | 96 |
| sliced_initial-native_fit-versus-baseline | gain_distance | 88 | 6 | 0 |
| sliced_initial-teacher_up4-versus-baseline | mae | 96 | 0 | 0 |
| sliced_initial-teacher_up4-versus-baseline | mel | 96 | 0 | 0 |
| sliced_initial-teacher_up4-versus-baseline | cosine | 94 | 0 | 0 |
| sliced_initial-teacher_up4-versus-baseline | group_mse | 96 | 0 | 0 |
| sliced_initial-teacher_up4-versus-baseline | quiet_rms | 68 | 0 | 0 |
| sliced_initial-teacher_up4-versus-baseline | quiet_passes | 68 | 0 | 28 |
| sliced_initial-teacher_up4-versus-baseline | near_passes | 15 | 0 | 81 |
| sliced_initial-teacher_up4-versus-baseline | gain_distance | 94 | 0 | 0 |
| trained4625-native_fit-versus-baseline | mae | 0 | 96 | 0 |
| trained4625-native_fit-versus-baseline | mel | 0 | 96 | 0 |
| trained4625-native_fit-versus-baseline | cosine | 0 | 94 | 0 |
| trained4625-native_fit-versus-baseline | group_mse | 9 | 87 | 0 |
| trained4625-native_fit-versus-baseline | quiet_rms | 1 | 67 | 0 |
| trained4625-native_fit-versus-baseline | quiet_passes | 0 | 29 | 67 |
| trained4625-native_fit-versus-baseline | near_passes | 0 | 13 | 83 |
| trained4625-native_fit-versus-baseline | gain_distance | 0 | 94 | 0 |
| trained4625-teacher_up4-versus-baseline | mae | 3 | 93 | 0 |
| trained4625-teacher_up4-versus-baseline | mel | 33 | 63 | 0 |
| trained4625-teacher_up4-versus-baseline | cosine | 88 | 6 | 0 |
| trained4625-teacher_up4-versus-baseline | group_mse | 68 | 28 | 0 |
| trained4625-teacher_up4-versus-baseline | quiet_rms | 19 | 49 | 0 |
| trained4625-teacher_up4-versus-baseline | quiet_passes | 4 | 24 | 68 |
| trained4625-teacher_up4-versus-baseline | near_passes | 0 | 13 | 83 |
| trained4625-teacher_up4-versus-baseline | gain_distance | 0 | 94 | 0 |
| trained-original-residuals-versus-adapted-residuals | mae | 96 | 0 | 0 |
| trained-original-residuals-versus-adapted-residuals | mel | 88 | 8 | 0 |
| trained-original-residuals-versus-adapted-residuals | cosine | 94 | 0 | 0 |
| trained-original-residuals-versus-adapted-residuals | group_mse | 96 | 0 | 0 |
| trained-original-residuals-versus-adapted-residuals | quiet_rms | 68 | 0 | 0 |
| trained-original-residuals-versus-adapted-residuals | quiet_passes | 23 | 0 | 73 |
| trained-original-residuals-versus-adapted-residuals | near_passes | 5 | 0 | 91 |
| trained-original-residuals-versus-adapted-residuals | gain_distance | 92 | 2 | 0 |
| trained-native-fit-original-residuals-versus-baseline | mae | 1 | 95 | 0 |
| trained-native-fit-original-residuals-versus-baseline | mel | 1 | 95 | 0 |
| trained-native-fit-original-residuals-versus-baseline | cosine | 0 | 94 | 0 |
| trained-native-fit-original-residuals-versus-baseline | group_mse | 96 | 0 | 0 |
| trained-native-fit-original-residuals-versus-baseline | quiet_rms | 5 | 63 | 0 |
| trained-native-fit-original-residuals-versus-baseline | quiet_passes | 3 | 21 | 72 |
| trained-native-fit-original-residuals-versus-baseline | near_passes | 0 | 11 | 85 |
| trained-native-fit-original-residuals-versus-baseline | gain_distance | 40 | 54 | 0 |

Directions retain every measured difference. Existing numerical reproduction tolerances are recorded separately in the JSON and are not perceptual acceptance thresholds.

## Mechanism and decision

1. **The original downstream decoder is sufficient when given the right internal signal.** Injecting the exact teacher up4 output into the sliced initialization recovers waveform MAE4.80e-9, all2,544 quiet passes and all184 near-silence passes. Injecting the teacher group-end into the unchanged suffix gives exactly zero waveform and mel error on all96 sources. These are controlled teacher-feature interventions, not deployable paths.
2. **The trained internal coordinates are no longer interchangeable with the original teacher's.** Injecting exact teacher up4 output into the adapted residual stack gives active cosine0.993828 but waveform MAE0.019783 and pooled active RMS1.663 times the teacher. All94 active recordings have excess RMS. The signed teacher-axis gain, sum(prediction×teacher)/sum(teacher²), is0.937 in the baseline,1.521 with native fitting,1.641 with exact up4 injection, and0.902 with native fitting plus original residuals. Exact local feature matching therefore does not imply correct final amplitude, and0.99 correlation alone would misleadingly approve this variant.
3. **The changed residual stack causally mediates much of this splice failure.** Keep the fitted native operator and all its upstream inputs fixed; restore only the three original stage4 residual units. MAE falls75.91%, from0.020350 to0.004903, with improvement on all96 sources. The pooled active RMS ratio falls from1.608 to0.939. Because this intervention changes only that residual stack, this is direct evidence of downstream coadaptation, not just an association between two loss curves.
4. **That control is not a quality fix.** Relative to the retained trained baseline, fit plus original residuals still has15.79% higher MAE,37.09% higher mel error and9.35% higher quiet RMS. MAE worsens95/96 sources, mel95/96 and active cosine94/94. Quiet passes fall138→89; near-silence passes18→5. Group-boundary MSE improves39.13% and every source, again demonstrating why that feature statistic cannot replace waveform quality.
5. **The native linear fit works mathematically, but its available student features remain an imperfect predictor.** Local up4 SSE improves86.93% at initialization and81.91% at the trained checkpoint, on all96 sources in each case. Both phases improve, including quiet and near-silence regions. At the trained checkpoint, held-out fitted MSE0.00149656 remains4.92 times the teacher-retained-input control0.000304315; the quiet gap is4.21 times and the near-silence gap9.93 times. Fit and development errors are comparable. Both actual-student solvers have full rank512, tiny normal-equation residuals, and passing folded-native parity. This is evidence of limited predictability within this fixed native linear operator, not proof that the upstream student has irreversibly lost the information or cannot learn a nonlinear alternative.
6. **This does not uniquely explain the preceding projected-hint A/B.** Those hints compared trainable readouts of student features with teacher features; they did not inject raw teacher features into the decoder. The present experiment establishes raw-splice risk and internal coadaptation. It does not identify the unique cause of the projected-hint regression or the earlier training trajectory's gain drift.

No runtime layer was added, no optimizer update was made, and no CPU timing was measured. A future recipe that preserves original downstream coordinates from its own starting point would require a separately approved training comparison. These results do not justify patching or restarting the accepted model automatically.
