# Complete-constraint direction test

11 September 2026. Authorized follow-on to the original and extended-grid retention pilots. Fresh GRAIL combination remains conditional on a useful retention method. This protocol changes no inference layer or teacher target.

## Fixed comparison

Recreate the original fresh-C run from original teacher-derived initialization, empty AdamW and original RNG. Replay the same 43 ordinary updates and require every recorded non-timing scalar to match the preserved original retention pilot. Restore the state immediately before the first rejected update. The actual Adam proposal, all 90 trainable group parameters and six training-calibration startup anchors are then fixed for both counterfactuals.

| Arm | Linear constraints | Nonlinear acceptance |
|---|---|---|
| Existing update | Worst residual and worst amplitude, two rows | All twelve original individual inequalities |
| Complete projection | Residual and amplitude for each of six anchors, twelve rows | The same twelve original inequalities |

Use the same extended fraction grid, 1 through 1/1024 by halvings, followed by exact zero fallback. Test actual FP32 parameter writes and original canonical waveform limits. A same-prediction FP64 metric remains diagnostic only. No threshold relaxation, additional loss, optimizer reset, alternative precision or development input may influence the projected direction or accepted fraction.

The complete projection minimizes distance from the same Adam displacement. Normalize each gradient row and its bound together. The small FP64 solve must report its rank, conditioning, active constraints and KKT/primal verification. Dependent rows cannot be silently dropped without verifying the full constraint system. A failure of the solve is a numerical outcome, not proof that the model lacks a feasible update.

## What is measured

- Accepted fraction, actual parameter displacement, correction magnitude and angle to the Adam proposal.
- All twelve predicted and actual constraint outcomes, including the maximum-switch and finite-step limitations already observed.
- Calibration startup passing, plus separately scored development startup passing and ordinary reconstruction objective for the same fitting batch at each accepted candidate. These observations do not select the direction using development data.
- Exact restoration of model, optimizer, gradients, RNG, frozen teacher and source files. No candidate weights are saved. Only aggregate statistics and provenance hashes leave Runpod.

This is a local direction experiment at original update 43. It is not a decomposition of every later rejected step in the extended-grid trajectory and does not by itself establish sustained learning.

## Conditional follow-on

If complete linear constraints produce useful feasible movement, validate that method in a fresh matched short recovery pilot before a full run. If the complete linear model passes but the true waveform still rejects the direction, test a bounded correction based on the actual rejected output excess in a separate arm. Keep the same real-waveform gate; do not bundle a new quiet weighting or architecture change into that correction.

A full 2,000-update trial requires useful ordinary reconstruction learning and retained startup behavior, rather than counting Adam updates that make no weight changes. Report the reused development panel and all quiet cohorts explicitly. A finite calibration set does not guarantee unseen startup behavior.

After the retention method is qualified, prepare a separate fresh original-teacher-derived GRAIL-like initialization and calibrate its startup response. Its starting state must be feasible before applying a preservation rule. Do not splice trained retention weights into the trained G endpoint. The frozen encoder, original latents and original waveform/mel/boundary targets remain common to the combined experiment.

Previous ordinary, R, extended-grid, B/C/D/G and failed-Q experiments remain preserved. Further width reductions remain deferred. The reasoning behind complete constraints and bounded nonlinear correction is recorded in [the primary-method review](feasible-update-review.md).
