# Startup-preserving native upsampler fit

The user authorized the follow-up to reconstructed A and B. This experiment is independent of the unchanged B recovery run and must not alter its initialization or training.

Start from the authenticated original teacher-derived384/256 slice. Refit only the stage3 upsampler with the same all-valid72-source calibration objective, delta ridge1e-6, five native phases, current/previous384-channel input, kernel10, stride5, causal trim5 and one shared bias.

Add equalities for the teacher upsampler outputs on complete8-sample cells within the original cached teacher's near-silent first20ms calibration windows. Verify that the native teacher selects the same windows. Do not reset context or use development samples for constraints.

Use centeredFP64 statistics and Cholesky-whitened projection. Fix rank relative cutoff to max(whitened constraint shape)*FP32 epsilon and absolute cutoff0 before examining development. Check compatibility against COMPLETE teacher outputs at the original interface tolerance (atol1e-5,rtol1e-4). Report the rank, discarded modes, correction size and actual residual. Do not chase nearly duplicate FP32 input rows with enormous coefficients.

Infeasibility is a valid diagnostic result: preserve the unchanged candidate, return an explicit no-stable-feasible-solution status, and do not tune ridge or tolerances. It would not prove that the width or architecture cannot reproduce the waveform; exact hidden equality is sufficient, not necessary.

If feasible, install the coefficients in the existing native operator, verifyFP32 writeback and equality residuals on calibration data, then evaluate the original96 development recordings with unchanged waveform/mel/group/quiet metrics. Separate startup from other near-silence and ordinary quiet audio. No new neural training, inference modules, automatic promotion, new cut, commit or push.

Keep audio, latent values, source identities, per-source diagnostics and fitted tensors on Runpod. Return only aggregate statistics and provenance checks.

