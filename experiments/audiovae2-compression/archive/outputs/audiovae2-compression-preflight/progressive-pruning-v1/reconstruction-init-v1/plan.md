# Reconstruction before recovery training

The user authorized this bounded experiment after the current-cut diagnosis and the aggregate-only startup comparison. Preserve the original step0 and step5000 checkpoints. Do not start another cut or long recovery run automatically.

## Purpose and scope

The original channel selector ranks stage-output features, then slices the existing weights. It does not refit the following operation to compensate for the inputs removed. Test whether a teacher-guided weight reconstruction provides a better starting point without changing the inference architecture.

All fits start from the authenticated fresh384/256 slice. Variant A refits only the stage3 upsampler. Variant B refits the three stage2 residual projections in sequence, then the stage3 upsampler. Recompute actual student inputs after each fit. Do not overwrite teacher-coordinate fits into the coadapted step5000 checkpoint.

For a residual projection, the target correction is the teacher's complete residual output on retained coordinates minus the current complete student residual output. The regressor is the actual student projection input. This includes the effect of the current student skip.

For the upsampler, the target correction is the complete teacher upsampler output minus the current student output. The regressor represents the five native phases and their current/previous384-channel inputs. Preserve native stride5, kernel10, causal trimming, and one shared output bias. Context precedes the scoring mask and must not be reset at crop boundaries.

Use centered FP64 sufficient statistics over all valid rows from the existing72 calibration sources. Use fixed delta ridge lambda =1e-6 times mean centered design variance, with an unpenalized shared intercept. No inverse-RMS weights, random row subsampling, development-based regularization tuning, or gradient training. Fold corrections into the existing effective weights and bias with the existing weight-normalization helper.

## Validation

First run focused CPU tests for native temporal geometry, masks/context, shared bias, residual skip handling, solver correctness and effective-weight installation. On Runpod use the preserved qualified runtime. Teacher inference remains frozen and uses the existing validated execution policy.

Evaluate fresh slice, A and B on the same96 development recordings with unchanged waveform, mel, full-group and quiet metrics. Report local fit errors alongside full-waveform outcomes. Compare with sealed step5000 as a recovery reference, not as the starting point of these fits. These96 recordings are development data disjoint from fitting, not a new untouched final test set.

Report active waveform correlation and error, mel error, full-group error, all quiet failures, startup failures, other near-silence failures, later quiet failures, quiet residual RMS and peak. Count individual recordings that improve or regress. Also report calibration startup and teacher-near-silence coverage. Do not equate fewer local feature errors with a successful waveform model.

Keep all original artifacts and teacher state unchanged. No additional inference operations or parameter dimensions are allowed. Keep fitted weights, raw diagnostics, audio and latent values on Runpod; return aggregate statistics only. Report measured results before deciding on recovery training or a gradual pruning transition.

## Literature connection

[StreamCodec2](https://arxiv.org/html/2509.13670v1) motivates intermediate teacher guidance.

[Channel Pruning](https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html) and [ThiNet](https://www.lamda.nju.edu.cn/luojh/project/ThiNet_ICCV17/ThiNet_ICCV17.html) provide the closer methodological precedent: select channels using downstream behavior and reconstruct the remaining operators. Applying that approach to this causal audio decoder is an experiment, not a published guarantee of silence parity.
