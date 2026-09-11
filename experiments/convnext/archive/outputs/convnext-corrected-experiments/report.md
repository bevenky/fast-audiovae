# Decoder experiment results

Targeted audio selection and discriminator views are the strongest changes to retain. Recalibrating changed losses corrects a real training problem, but restarting a magnitude discriminator did not improve the best continuation. The complex multiband discriminator is a promising spectral-quality candidate with remaining quiet-audio and overshoot tradeoffs. No candidate fixes every issue.

All four arms completed 400 generator updates from the same immutable step-8,090 checkpoint, finishing at step 8,490. Each saw 12,800 windows, or 8.978 hours of valid scored audio. The AudioVAE2 encoder and teacher decoder stayed frozen. Student architecture, model-normalization statistics and initial generator optimizer state were preserved. There were no new student parameters or inference operations.

## Quality comparison

These are teacher-relative measurements on the same 282 natural crops from 144 sources. Lower error, quiet RMS and overshoot counts are better. Quiet RMS is shown multiplied by one million for readability. Overshoot counts are scored samples above absolute amplitude 1; overlapping evaluation crops mean these are not counts of unique physical events.

| Model | Waveform MAE | Mel error | Quiet RMS × 10⁶ | Maximum peak | Overshoot samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original step 8,090 | 0.012814 | 1.1561 | 346.1 | 1.2535 | 420 |
| Regular continuation | 0.012692 | 1.1638 | 432.1 | 1.2890 | 471 |
| Targeted data and views | 0.012552 | 1.1354 | **306.7** | 1.2655 | 457 |
| Targeted, fresh magnitude discriminator and calibration | 0.012596 | 1.1420 | 322.5 | **1.2482** | 458 |
| Targeted, fresh complex discriminator and calibration | **0.012546** | **1.1019** | 312.2 | 1.2634 | 454 |

Waveform MAE is sample-weighted; mel error uses an unchanged diagnostic criterion averaged over crops. The complex arm reaches 0.9203 mean nonquiet waveform cosine, compared with 0.9129 initially. This remains below the final reconstruction goal.

**Targeted data and views:** compared with regular continuation, natural quiet RMS falls 29.0%, expressive waveform MAE falls 3.0%, and expressive mel error falls 3.6%. Speech waveform MAE is effectively unchanged, with mel error 1.8% lower. Compared with the original checkpoint, natural quiet RMS falls 11.4%, so the benefit is not merely avoiding the control's regression. The source-resampled interval for its natural MAE change versus regular is −1.93% to −0.45%; this does not establish perceptual equivalence or training-seed robustness.

**Fresh magnitude discriminator:** compared with the targeted arm retaining its trained discriminator, natural MAE worsens 0.35%, mel error worsens 0.58%, and quiet RMS worsens 5.16%. It reduces the largest peak but does not establish a useful overall advantage. Do not select this restart as the preferred continuation.

**Complex discriminator:** compared with the fresh magnitude arm under the same preparation budget, mel error improves 3.51% and quiet RMS improves 3.20%. Against the stronger targeted arm retaining its discriminator, mel error improves 2.95% and high-frequency magnitude error improves 5.99%, while waveform MAE is effectively tied. Natural quiet RMS worsens 1.79%. Overshoots affect eight crops from six sources, versus seven crops from five sources for targeted training. The extra affected source has one sample at approximately 1.0042; the larger existing overshoots remain the more serious problem.

## Expressive cases

These are small, source-labelled groups, not timestamp-level verified event annotations. Correlation below means aligned waveform cosine on nonquiet crops.

| Held-out group | Sources | Regular correlation | Targeted correlation | Complex correlation |
| --- | ---: | ---: | ---: | ---: |
| Breathing | 3 | 0.591 | 0.627 | 0.636 |
| Laughter | 6 | 0.879 | 0.881 | 0.881 |
| Screaming | 7 | 0.697 | 0.739 | 0.742 |
| Whispering | 4 | 0.721 | 0.730 | 0.730 |
| Human whistling | 2 | 0.672 | 0.718 | 0.732 |
| Yelling | 1 | 0.950 | 0.951 | 0.951 |

Whistling and screaming respond to targeted exposure. Laughter remains a tradeoff: targeted training improves average reconstruction but raises its maximum peak from 1.1335 in regular continuation to 1.1675, with 119 overshoot samples instead of 95. Complex reduces that peak to 1.1489 but still leaves 115 overshoot samples.

Training and validation include all 22 scheduled Indic language codes. The natural evaluation contains 32 language-label groups, including an undetermined-language group. Coverage per language is small. Targeted versus regular MAE regresses on Tamil by 1.75%, Telugu by 1.67% and Sindhi by 1.24%, each represented by three sources. There are no distinct held-out crying, giggling or separately reviewed shouting groups, so these results do not validate every requested expression.

## Calibration and the silence diagnostic

The fixed-weight calibration corrected stale gradient statistics without updating student parameters or optimizer state:

| Fresh discriminator | FM + adversarial gradient share before calibration | Immediately after | Mean during 400 updates |
| --- | ---: | ---: | ---: |
| Magnitude | 8.80% | 29.99% | 36.12% |
| Complex | 0.18% | 30.44% | 40.20% |

These are proportions of output-gradient norms before summing, not parameter-update shares. The intended combined proportion is 30%. Fresh discriminator gradients subsequently grow faster than their moving averages, increasing effective influence. In the complex arm's final fixed audit, FM and adversarial norms are 1.82× and 1.95× their EMA estimates. There is no scale saturation in calibration or the fixed before/after audits, and the measured combined quiet-gradient direction remains aligned with reducing teacher residuals.

Therefore calibration is a valid correction when changing losses, but its independent quality benefit was not isolated: fresh arms also change discriminator initialization and add 64 discriminator-only updates. Likewise, the complex comparison is at equal nominal settings and preparation budget, not equal achieved gradient proportions throughout training.

The frozen teacher gives identical outputs on repeated decoding of the same encoded-zero trajectory. Its difference from the cached target is only 6.54e-9 maximum absolute amplitude. Teacher instability does not explain the student residual.

Initially, approximately 91–94% of residual power on the encoded-zero and quiet-noise fixtures follows a 480-sample repeating pattern. Ordinary adjacent-sample jumps are similar to block-boundary jumps on those fixtures. This does not support a boundary-only explanation or prove which layer causes the error.

Targeted training lowers encoded-zero residual RMS from 2.210e-4 to 1.422e-4. Complex lowers it further to 1.275e-4, but its extra improvement comes mainly from reduced DC error. Its 480-sample AC residual power is 18.5% higher than targeted training and accounts for about 87% of its remaining residual power. Neither candidate has removed the repeating waveform.

**All 3,434 natural quiet windows and all 639 quiet windows across the synthetic fixtures still fail the existing strict checks.** Improved RMS is progress, not a completed silence fix. None of the candidates qualifies as a solution to peak overshoot either.

## Decision and verification

- Retain targeted data and selected discriminator views as the preferred training recipe, while tracking language and laughter regressions.
- Retain the calibration implementation for changed objectives. Do not infer that resetting a trained discriminator is beneficial.
- Preserve the complex checkpoint as a spectral-quality alternative for review. Keep the retained-discriminator targeted checkpoint beside it; there is no unqualified winner between them.
- Preserve the residual diagnostics. They narrow the remaining issue to structured error but do not justify adding a new output layer on their own.

No candidate has been promoted, committed or merged. The original long run remains paused, and all four final checkpoints are separate on Runpod.

Independent verification checked all 1,600 contiguous generator updates, planned sample counts and discriminator views, parent and final model fingerprints, frozen normalization, calibration receipts and unchanged evaluation panels. All 16 CPU streaming checks passed at 80 and 160 ms, with exact output counts and maximum absolute batch/stream difference of 8.65e-7, below the 2e-6 tolerance. These are CPU correctness checks on Runpod, not fresh cross-platform RTF benchmarks.

Generator training took 122–123 seconds per magnitude-based arm and 153 seconds for complex on the H100, excluding teacher preparation, warmup, calibration and evaluation. Targets were prepared and cached on Runpod. No training audio was downloaded to the Mac. This is one 400-update trajectory per arm, with no new listening, PESQ, STOI or MOS assessment.

The [experiment plan](plan.md), [data selection](data-selection.md), [verified results](results.json) and [all event and language results](event-language-results.json) contain the supporting measurements and provenance.
