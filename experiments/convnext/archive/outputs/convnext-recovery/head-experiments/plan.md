# Targeted decoder-head experiments

The layer diagnosis supports testing the existing final synthesis mapping before adding a new temporal network. All ten ConvNeXt blocks, the encoder, teacher, normalization and upstream student weights stay frozen. Every candidate starts from the same step 8,890 checkpoint. No full training run or production replacement is implied.

## 1. Stationary silence

The teacher develops phase structure internally but its learned output convolution largely cancels it. The student's existing final features can represent the teacher's steady response. Therefore first change the existing readout coefficients, adding no inference operation:

- A shared-pattern correction matches the teacher's average 480-sample block while preserving current differences between the four internal phases.
- A full correction matches all four stationary phase blocks.

Both are minimum-norm linear constraints on the readout, not waveform subtraction or a silence detector. They are calibrated on a freshly encoded eight-second zero recording, using the actual encoder means and teacher waveform. The existing six-second fixture is explicitly the same synthetic signal family; success there is constraint verification, not unseen-audio generalization.

## 2. Natural quiet audio

Most natural-quiet error changes with the input. Matching one silence response cannot establish a solution. Fit the final readout to real teacher targets while satisfying the silence constraint. Use regularized least squares on the actual frozen student features. This is a convex, precisely defined head-only distillation problem, with an analytic solution rather than a new optimizer or a loss-weight guess.

Use 2,048 unique training recordings for fitting and 256 different recordings for regularization selection, with audio-hash and source exclusion against the 147-source canonical panel. Each fitting recording contributes one crop. Complete valid 480-sample frames enter the fit; final evaluation retains every sample in the original evaluation mask.

Fit four fixed trace-scaled ridge strengths: 0.001, 0.01, 0.1 and 1. Select using the separate training-validation split only: no more than 1% increase in waveform MAE or teacher-defined quiet MSE, then lowest waveform MSE among qualifying choices. No qualifying choice means that family fails this screen. A constrained head fit is not assumed to preserve perceptual quality simply because its training MSE improves.

## 3. Loud transients

The teacher's final tanh actively compresses internal peaks near 3 into a bounded output. The current student already predicts the teacher's compressed waveform, so directly inserting tanh alters valid predictions.

Test a learned tanh-head migration using the teacher's actual pre-tanh convolution output as the fitting target. Capture that tensor from the frozen teacher, verify its ordinary output against cached targets, and apply the same constrained head fitting and independent validation procedure. Final quality always uses the teacher's ordinary post-tanh waveform. Never infer targets by taking inverse tanh of rounded samples.

An exact clamp to [-1,1] is included only as a diagnostic bounding control. It preserves in-range samples and cannot increase pointwise error against bounded teacher samples, but it can alter spectral or transient quality. Zero overshoot after clipping does not prove the learned model was repaired.

## Comparison and decisions

Evaluate the unchanged head, shared correction and full correction both raw and with the clamp control. Evaluate each learned head family only if its training-validation screen qualifies. These alternatives replace one existing weight matrix; only clamp/tanh adds a pointwise inference operation. No additional state or lookahead is introduced.

Use the complete corrected 285-crop, 147-source panel with unchanged teacher-defined quiet masks and scoring exclusions. Report waveform, mel, high-frequency and first/second-difference errors, expressive/speech/language groups, stationary silence, natural quiet and peaks. Keep the existing engineering screening margins: natural waveform/mel within 1%, speech/expressive within 2%, represented language/event groups within 5%. A silence improvement does not qualify a model that violates those margins. These are screening limits, not audibility thresholds.

Apply the same 1% natural and 2% speech/expressive margins to high-frequency magnitude/complex error and first/second-difference RMS error. To qualify as addressing all three problems, a candidate also needs at least 10% lower natural-quiet RMS, lower stationary-silence error and no output above full scale. Partial successes are reported as partial, especially the clipping control.

CPU checks use one thread and no GPU. Verify 80/160 ms streaming sample accounting and numerical parity, then compare the added pointwise activation cost. Unchanged matrix dimensions and operation counts are not presented as a measured production RTF speedup.

Preserve the original checkpoint and optimizers. Save isolated head candidates and evidence. Do not promote a candidate automatically or tune it using canonical evaluation results.

## Interpretation limits

These experiments test the cheapest existing synthesis mapping before proposing a larger architectural change. The teacher also learns cancellation through its synthesis weights; the anchor follows that mechanism without adding a silence detector. It does not reproduce the teacher's multichannel temporal convolution. Its equality constraint remains valid only while the fitted readout and upstream features stay fixed.

The real-audio fit uses sample-pooled squared error and a penalty toward the trained readout. It does not preserve the original adversarial or mel objective automatically. The pre-tanh fit is not the nonlinear optimum of post-tanh waveform error. Failure under the fixed constraints and regularization grid would reject this particular calibration method, not prove that the upstream architecture lacks capacity.

The canonical panel has been inspected repeatedly during development. It is kept separate from fitting and regularization selection here, but is a development validation panel, not a new independent final test. Passing its engineering screens would justify a further quality review, not a claim of perceptual equivalence.

## Teacher capture qualification

The first attempt stopped before fitting any head because isolated-crop decoding did not exactly reproduce one saved whole-recording target. A separate repeated-call probe then showed that the first encoder execution differed slightly, while calls 2 through 4 exactly reproduced the cached latents. Every decoder call using those latents recreated the original whole-source cache key. This establishes execution-startup dependence for that example; it does not isolate a particular compiler or backend as the cause.

The revised experiment records four startup encoder calls and requires the last two to agree. Teacher head targets are captured using the original authenticated full-recording geometry for every fit source. The entire cached latent/output crop and original full-source cache key must match exactly. No tolerance, cached target or evaluation mask is changed. The failed first attempt remains recorded separately.
