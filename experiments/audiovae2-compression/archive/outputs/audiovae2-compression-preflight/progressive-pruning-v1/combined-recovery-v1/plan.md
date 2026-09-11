# Combined reconstruction and startup recovery

## Question

Can a fresh teacher-derived student combine B's reconstruction of the four affected native operations with the startup-constrained upsampler fit, then recover ordinary waveform quality over 2,500 updates while retaining startup behavior?

This is a training experiment, not a promotion of the untrained constrained-A candidate. It does not separately measure how constrained A alone would train.

## Initialization

1. Recreate the same stage-2 channel cut from the original AudioVAE2 teacher weights: 512 to 384 channels. Keep the existing stage-3 width and all residual units.
2. Install the sealed B reconstruction operators. Verify the full starting decoder state matches the preserved B initializer.
3. Recompute the stage-3 upsampler inputs after B's stage-2 changes.
4. Refit that upsampler with the exact startup-constrained solver and calibration selection used in constrained A. Retain the other three B operators.
5. Verify native FP32 execution against the fitted equations, calibration constraint feasibility, unchanged teacher and outside parameters, and a finite full evaluation.
6. Save the four compact native operators, provenance and full decoder state hash. Do not copy A's fitted weights or use any trained 1,000/2,000/5,000-step student to initialize.

The fixed 72 calibration sources fit the initialization; the separate 96 development sources measure it. No ridge, rank cutoff, data selection or loss setting is chosen from development scores. Failure to fit the specified calibration constraints is a method failure, not permission to silently replace the candidate with ordinary B.

## Recovery

- Original frozen encoder, teacher decoder and student stages outside 2 through 4.
- All 90 parameter tensors in stages 2 through 4 train jointly.
- Original 64-channel latent inputs and original cached/live-checked teacher targets.
- Fresh AdamW: learning rate 0.00003, betas 0.9/0.99, epsilon 1e-8, weight decay 0.
- Physical batch 1, gradient accumulation 12. Original qualified FP32 training runtime and numerical policy.
- Original waveform, mel and group-output loss coefficients. No new quiet loss or optimizer projection in this experiment.
- Exact original step-0 RNG restored after setup.
- Original executed source order through 30,000 distinct sources. The first 24,000 match B's completed 2,000-update comparison. No within-run reuse.
- Reviews at 0, 250, 500, 1,000, 1,500, 2,000 and 2,500.
- Save model group, optimizer, RNG and data ledger at 0, 1,000, 2,000 and 2,500.
- Hard stop at 2,500. No automatic continuation or next cut.

A passing initial startup response may be changed by these unconstrained optimizer updates. Measuring that behavior is part of the experiment. A change that preserves startup throughout optimization would be a separate training-method comparison.

## Comparison and decision

Compare combined recovery with B at equal available steps through 2,000, and with the original sliced student at equal steps including 2,500. Keep the old 5,000 result as a reference with more training exposure, not an equal-budget control.

Report active-waveform correlation, absolute waveform error, mel error, group-output error and peak overshoot. Separate all 13 startup windows, the other near-silence windows, and remaining quiet windows. Keep fixed membership and thresholds. Report continuous residual/output levels beside pass counts.

Retain the endpoint even if an earlier checkpoint scores better. No intermediate checkpoint selection, silent restart or hyperparameter change. No inference operations are added; CPU RTF is not rebenchmarked by this recovery experiment.

## Operation and provenance

All data, targets and model artifacts stay on Runpod. Only aggregate statistics, integrity booleans and non-sensitive provenance return locally. Preserve the original teacher and all previous checkpoints.

The combined initializer and recovery use separate output roots on the Runpod workspace volume. Code, tests and launch parameters will be sealed before execution. TensorBoard will use a separate event directory and publish the actual completed reviews.

The prior B comparison completed 2,000 updates on 24,000 unique sources in 21.83 minutes of core run time. Its independent completion audit passed. It improves some same-step metrics, but does not replace the old 5,000-step model and still has startup failures.
