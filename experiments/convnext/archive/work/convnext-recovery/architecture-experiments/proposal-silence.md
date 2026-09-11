# Stationary silence: minimal intervention proposal

The teacher's final 32-channel, seven-tap convolution cancels periodic components. The student's final 480-by-2048 matrix can already perform the required cancellation on the observed stationary features. Copying the teacher's layer would therefore add computation without establishing a missing capacity. The first experiment should alter how the existing readout handles the measured silence feature subspace, preserving the ten useful ConvNeXt blocks.

## Evidence and distinction

- Current stationary teacher residual RMS is 4.3793e-5.
- Shared 480-sample structure plus DC accounts for 98.6624% of residual power. The remaining four-phase contribution is 1.3376%.
- Four actual final feature vectors form a rank-four matrix, condition number 65.31. Matching all four teacher blocks is feasible with the current matrix, although the minimum weight change is 2.1686% of its norm and has not been shown safe for natural audio.
- Teacher tanh is exactly identity on this fixture in FP32. It supplies a peak bound, not the measured silence cancellation.
- Natural quiet is different: 81.93% of its previously measured residual varies over time. A fixed waveform-template subtraction improved quiet RMS only 0.65%.

The mechanism agrees with the [Pons et al. paper](https://arxiv.org/abs/2010.14356) and [authors' subpixel analysis](https://github.com/DolbyLaboratories/neural-upsampling-artifacts-audio/blob/main/ARTICLE.md#4-subpixel-convolutions): different synthesis weights at successive output positions permit tones, including after training. Initialization changes alone do not guarantee they stay absent. This supports an explicit learned cancellation test, not globally sharing every output row or low-pass filtering all audio.

## Candidate A: constrain only the existing silence response

Let H have shape 2048-by-4: the four phase features from the same steady encoded-zero latent fixture, after 2 seconds. Let W be the current 480-by-2048 output weight and T the corresponding four teacher waveform blocks, shape 480-by-4. All calculations for constructing the update use FP64; the installed disposable weight uses the model's original FP32 type.

### A1: dominant-component correction

Define e = mean_phase(T - W H), shape 480-by-1. Require the same correction at each of the four phases:

    Delta W H = e [1, 1, 1, 1]
    Delta W = e [1, 1, 1, 1] H^+

This preserves the differences between the four current phase outputs while correcting the shared 480-sample component and its DC. It changes one rank-one direction in the readout. It does not add a fixed waveform to every input: its effect on ordinary input features h is Delta W h. The actual scope of that effect must be measured.

The simpler one-vector correction Delta W = e h_mean^T / ||h_mean||^2 has an estimated weight norm of about 0.31% of the existing readout, using the recorded mean-feature norm. That estimate does not apply exactly to the phase-difference-preserving constrained version above; compute and report its actual norm rather than claiming the estimate as a result.

### A2: full four-phase correction

    Delta W = (T - W H) H^+

This matches all four stationary blocks. It is an exact linear-algebra solution for this fixture, not a general quality guarantee. The existing rank test estimates a 2.1686% weight norm change. Test A1 first; A2 is a stronger comparison, not automatically the better choice.

### Preserve ordinary audio in the fit, if the minimum-norm fit regresses

A predetermined positive-definite feature covariance C = X X^T / N + lambda I from a disjoint representative training calibration set can replace the Euclidean norm. The least output-disturbing constrained correction under that metric is:

    Delta W = E (H^T C^-1 H)^-1 H^T C^-1

Here E is either repeated e (A1) or T-WH (A2). Solve systems; do not explicitly invert C. This minimizes calibration output displacement subject to the selected silence condition. It does not minimize held-out reconstruction error and should not be tuned after seeing held-out results. Avoid adding this extra arm until the simple intervention demonstrates whether preserving natural audio is actually a problem.

## Candidate B: trainable readout parameterization, no new inference operations

For a bounded continuation with the body frozen, retain the silence constraint after every readout update:

    W_after = W_trial + (T - W_trial H) H^+

Equivalently parameterize W = T H^+ + B (I - H H^+). Every B remains free on the orthogonal complement of the four-dimensional silence span. This isolates the known cancellation from updates intended to improve natural audio. It does not discard or gate low-level input. The teacher's actual nonzero response is retained rather than forcing arbitrary zero latents to decode to zero.

Use A1's adjusted phase targets if selecting only the dominant-component condition. If combining with terminal tanh, the linear constraint must target atanh(T), with a declared finite clamp, rather than T; quiet values happen to be nearly unchanged, but the equation should remain correct.

This constraint is valid only while upstream features are frozen. If the body trains, H changes and the anchor must be remeasured from the same encoded-zero fixture after each update, or the guarantee no longer holds. That extra training machinery is unnecessary for the first head-only comparison.

The parameterization can start function-preserving by initially using T0 = W_original H, then changing the desired anchor in the disposable experiment. There is no lossless change from the original erroneous silence response to the desired silence response: every corrective candidate necessarily changes outputs somewhere. State this explicitly.

## Invariants and evaluation

- Original encoder, raw 64-channel latents, normalization, adapter, ten blocks, causal head history, PReLU and checkpoint remain fixed.
- No parameters or per-stream buffers are added. Export folds the result into the same weight matrix. Inference MAC count and graph structure are unchanged. This is not a new RTF benchmark result, but the arithmetic contract is exactly unchanged.
- Evaluate on canonical stationary silence, all existing natural-quiet cases, ordinary speech, loud speech and expressive sounds. The fitted stationary fixture is a calibration success, not held-out evidence.
- Measure phase-component residual, natural-quiet RMS, waveform and mel error, peak error, source-level regressions, and a silence-to-speech transition. A good average cannot conceal a large source regression.
- All fitting uses the designated training-only calibration sources. Held-out natural cases are scored after the candidate is fixed. No source, window or regularization selection from their outcome.
- Check batch/stream equality under the unchanged graph, output finiteness, original-state immutability, and FP32 versus solved constraint residual. A mathematical exact equality can become a small nonzero FP32 residual.

## Alternatives considered

An invertible decomposition of W into a common output row plus zero-mean phase rows can preserve the initial function and allow separate training step scales. Alone it changes neither capacity nor the silence response, so a claimed quality benefit would still require the above constraint or a different objective.

A two-channel waveform head followed by a 2-to-1 seven-tap convolution would provide more learned temporal mixing than the previously tried single-channel filter, but adds about 98.3 million MAC per second, roughly 3.9% of the current analytical total, before memory and filter costs. It still permits periodic artifacts, and is not presently justified by a representational failure. It is a later option only if head-only fitting cannot improve held-out data.

Sharing all 480 output rows or removing four-phase structure globally would erase synthesis freedom needed for ordinary voiced audio. A universal stationary subtraction does not solve input-dependent quiet errors. A zero-latent or amplitude gate violates the observed nonzero encoded-silence contract. None is a justified initial candidate.

Recommendation: compare untouched control, A1 and A2 as disposable head edits before spending training time. If a constrained readout preserves other cases, continue only the selected head with a frozen body and the already justified smaller output-displacement regime. If it fails natural quiet, retain the checkpoint and report that the cancellation is entangled with useful features; do not call the architecture fixed because the fitted silence is exact.
