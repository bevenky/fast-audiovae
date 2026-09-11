# Fresh GRAIL and startup-retention combination

11 September 2026. The bounded-correction retention pilot has qualified: all 64 updates moved, all six calibration starts stayed passing, development finished at 12/13, and ordinary waveform error matched unconstrained recovery within 0.004%. The fresh G-plus-startup initializer is now running. The user has authorized a fresh 2,000-update recovery after the matched short qualification, documentation of the method, and later investigation of deeper cuts. No combined recovery run has started yet.

## Initialization

Build the original frozen AudioVAE2 teacher and create the declared 384/256/128 student from its original decoder weights. Install the four authenticated, untrained GRAIL-like native operator fits from the preserved initializer artifact. Do not load the trained G endpoint, any other trained checkpoint, or the old combined initializer's saved group weights.

Using the same 72 calibration crops, capture the actual G-initialized input features at the stage-3 upsampler and the corresponding original teacher output. Apply the existing native reconstruction and constrained-startup fit using unchanged tap geometry, valid-cell weighting, shared bias, ridge and rank policy. Preserve all three G residual mixers exactly. Install the fitted operator in the same native upsampler; no layer or inference branch is added.

This operator refit can leave G's shared zero-intercept feature-map family and change its shared bias. Name the result a G-plus-constrained-upsample initializer. It is not an unchanged G artifact, and its difference from G includes ordinary reconstruction refitting as well as startup constraints.

Authenticate the original teacher, selected channels, four original G operators, three unchanged residual mixers, fitted upsampler, complete initial state, calibration membership, source/target cache and RNG. Verify the actual FP32-installed decoder on all six calibration starts. The thirteen development starts and full 96-source panel remain evaluation-only. Architectural interface equality alone is not a waveform quality pass.

## Matched qualification

Run two fresh 64-update pilots from this exact new initial state, with empty AdamW and identical original RNG, objective and 768 distinct ordinary sources:

| Arm | Only training-policy difference |
|---|---|
| Ordinary control | Original Adam recovery |
| Retention | The separately qualified startup-preservation rule, with six recurring calibration anchors accounted separately |

Both retain all 90 trainable group tensors, frozen teacher/encoder/outer stages, original latents and target geometry. Compare actual accepted movement in the final interval, objective improvement, ordinary waveform and mel errors, startup counts and continuous worst errors, all seven quiet cohorts, transients and peaks. Do not require near-final 0.99 correlation as a short-pilot entry condition. Do not interpret changing-batch losses as a fixed-panel trend.

Existing G2k remains a useful historical quality reference, but only the matched ordinary-versus-retention pair isolates the new recovery policy. Passing six protected calibration examples cannot guarantee all unseen startup windows, and C's qualification cannot be transferred without testing this initializer's actual features.

## Full experiment

If the matched pilot supports useful retention, start the declared bounded 2,000-update comparison afresh from this authenticated initializer, not from either pilot's trained weights. Keep the same 24,000 unique ordinary-source exposure contract and separately report recurring calibration involvement. Preserve all prior experimental artifacts and failures. Update the active dashboard and any existing follow-up only when this full experiment actually starts; the old failed queue remains stopped.

No automatic extension, additional width cut, inference benchmark, deployment promotion or threshold relaxation is part of this plan. Final quality and CPU RTF remain separate qualification work.
