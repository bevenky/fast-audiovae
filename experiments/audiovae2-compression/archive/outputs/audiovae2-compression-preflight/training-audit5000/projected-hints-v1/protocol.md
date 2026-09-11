# Projected feature hints: controlled recovery comparison

The approved experiment tests whether intermediate teacher guidance helps the preserved, trained AudioVAE2 student recover information lost by narrowing stages 2 and 3. This is a bounded diagnostic, with no further pruning or automatic promotion.

## Fixed comparison

Both arms start from the saved accumulation-12 candidate at AdamW step 4,625, including its student weights, optimizer moments and RNG state. Authenticate the checkpoint against the previous comparison receipt before use.

Each arm uses the same ordered 1,500 diagnostic sources, indices 10,500 through 11,999 of the sealed source plan. Those recordings were used in the previous diagnostic. This is intentional reuse while debugging, not a claim of fresh training exposure. Each source appears once within each arm. Preserve calibration/development separation and verify all required shards before updating weights.

Run 125 updates per arm, accumulating 12 singleton forwards with the existing sample-pooled reductions. Keep AdamW at learning rate 0.00003, betas (0.9, 0.99), epsilon 0.00000001 and zero weight decay. Keep the saved waveform, mel and full-stage-4 feature coefficients unchanged. Evaluate the fixed 96-recording development panel at the common start and fixed endpoint, with the existing four-case diagnostic panel every 60 sources. Do not select a favorable intermediate checkpoint.

| Arm | Training objective |
|---|---|
| Baseline | Existing waveform L1, mel and raw stage-4-end feature MSE |
| Projected hints | Same objective, plus stage-3-up and stage-4-up projected teacher feature MSE |

## Auxiliary feature guidance

| Location | Student | Teacher | Valid waveform samples per feature cell |
|---|---|---|---|
| Stage-3 upsampler output, before residual units | 128 channels, 6 kHz | 256 channels, 6 kHz | 8 |
| Stage-4 upsampler output, before residual units | 128 channels, 12 kHz | 128 channels, 12 kHz | 4 |

Capture the student's actual propagated features, not separately teacher-forced intermediate stages. Capture teacher targets under no-grad. Use per-frame affine projections, with no temporal mixing. The projections feed auxiliary losses only. The decoder, its frozen suffix, and export never consume projected features. The existing complete stage-4-end 128-channel target remains unprojected.

Initialize only the projections using the 72 training calibration sources and weighted ridge fitting. Use the fixed ridge factor 0.000001 times the mean diagonal of the centered input covariance. Preserve the student and teacher weights during calibration. Exclude context and count partial tails by the exact number of valid waveform samples. Do not normalize quiet examples by their audio RMS.

After initialization, train the projections alongside the student using a separate fresh AdamW optimizer with the settings above. This allows the alignment to adapt to the student's changing basis without changing the student's saved optimizer state. Retain a copy of the initial projection maps for endpoint diagnostics: improved projected loss alone must not be mistaken for improved student audio.

## Fixed hint weights

On the same 72 calibration sources, pool gradients with the existing valid-sample reductions. For each hint, set its coefficient so its gradient norm is 5% of the existing combined objective's gradient norm, measured only on student parameters upstream of that hint. Exclude projector gradients. Abort if a required norm is zero or nonfinite. Freeze these coefficients before the A/B run.

This is an experimental strength, not a published universal coefficient. Report both hint/reconstruction gradient cosines and the combined student-gradient ratio. A 10% aggregate raw-gradient budget does not guarantee a 10% change to AdamW's update with saved moments. Track actual student displacement where available, without adding an optimizer search.

## Checks and interpretation

Before execution, verify detached teacher targets, gradient flow into the intended student layers, exact mask reductions, identical decoder output with auxiliary capture enabled, and unchanged decoder state keys/shapes. Both arms must restore the same checkpoint, optimizer and source ordering. Preserve original checkpoints and code receipts; write results in a new directory.

Judge the fixed endpoints on waveform error, active correlation and level, complete-block error, mel error, silence/quiet residual, passing quiet windows, onset/transient behavior and expressive examples. Record per-source regressions as well as pooled means. Keep the final reconstruction goal of 0.99 active correlation alongside the other quality requirements; correlation alone is insufficient.

A favorable short result justifies a longer matched recovery test. An unfavorable or mixed result does not establish that all projected distillation fails. Neither outcome justifies further pruning or a release by itself. The deployed operation graph stays unchanged, so this experiment adds no decoder inference operators; it does not establish a new measured RTF.
