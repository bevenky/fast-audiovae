# AudioVAE2 and student architecture audit

Read-only architecture analysis, 2026-09-09. No training, model inference, benchmark, source modification or deployment was performed.

## Exact evidence

Pinned teacher source: `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`, verified by the loader against SHA-256 `2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8`.

[Pinned source](https://raw.githubusercontent.com/OpenBMB/VoxCPM/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py): `encode` returns `mu`; decoder has six upsampling stages, per-stage sample-rate scale/bias conditioning, Snake residual processing and final seven-tap causal convolution plus tanh. Causal convolutions use left zero padding. Default configuration has noise injection disabled. Weight normalization parameterizes the convolutions. The source exposes inference but does not establish a complete AudioVAE2 training recipe.

Student evidence in `experiments/convnext/audiovae_student/model.py`: raw-repeat phase adapter lines 121-135; 64-channel/25 Hz/48 kHz/1,920 sample contract lines 207-213; ten low-rate ConvNeXt blocks and unbounded 480-phase waveform projection lines 215-266; replicate startup padding lines 75-89; folding implementation lines 309-343. Recipe v2 requires the raw-repeat adapter (`recipe_v2.py:101`) and at least 29 real latent context frames (`recipe_v2_pilot.py:140-149`).

Teacher wrapper in `teacher.py` explicitly fixes raw posterior means, 48 kHz bandwidth conditioning, frozen FP32 evaluation, source/checkpoint hashes, no loudness normalization or latent sampling, whole-utterance encoding and original-length trimming. These are already the required teacher/student fusion contract. A new KL loss, latent rescaling, stochastic latent draw or new sample-rate embedding is not needed to preserve this deployed contract.

## Receptive-field calculation

Analytical integer support propagation, not a timing or model experiment: for each of the 1,920 output positions in one latent frame, subtract the final six waveform-sample history taps. In reverse decoder-stage order, subtract 78 residual-history samples and map through a stride-s transposed convolution using `floor(position/s)-1`. Subtract the six initial latent-history taps. This gives 20 preceding latent frames for 270 output phases and 19 for 1,650 phases. The maximum teacher decoder dependency is 20 preceding latents (0.8 seconds in latent-frame time).

The student has `6 + 6 * sum(1,2,4,1,2,4,1,1,1,1) + 2 = 116` internal history frames, mapping to 29 preceding raw latents (1.16 seconds). Insufficient structural decoder context is therefore not an obvious gap. This says nothing about effective learned use of context, or the encoder receptive field.

## Bounded recommendations

1. **Output tanh is a justified first candidate**, supervised against the current waveform targets during fine-tuning. It changes no latent or causal state contract. It is only 48,000 scalar evaluations per audio second. CPU cost remains unmeasured. Adding it to the current waveform without fine-tuning would compress loud speech: `tanh(0.9) = 0.716`. Near zero it is approximately the identity, so it does not remove residual silence noise.

   Tanh structurally bounds finite outputs to [-1,1]; finite precision can still return exactly +/-1. Report `abs > 1` as overshoot separately from `abs >= 1` saturation, retaining waveform error, loudness, crest factor and listening checks. A lower violation count alone is not evidence of improved reconstruction.

2. **Startup padding is a real difference, not a proven cause.** Zero padding would be more faithful to the teacher boundary convention, but do not flip it in a trained checkpoint without a controlled comparison. The student's exact final-affine folding currently relies on replicate padding: the folded bias assumes the affine offset extends into left history. Zero padding would require startup bias correction or an appropriate state representation to preserve exported parity. This difference cannot explain steady silence after the finite student history has elapsed.

3. **If 480-phase residual artifacts persist, consider local waveform coupling separately.** The student predicts 480 contiguous samples with distinct output channels. A shared, identity-initialized one-channel causal seven-tap output convolution would be a very small ablation inspired by the teacher's final local synthesis stage: 336,000 MAC/s and six samples of state. It can start as an exact identity before tanh. It is not equivalent to the teacher's multi-channel synthesis convolution and is not a proven fix; do not add it together with tanh by default. It could suppress useful high-frequency detail, so phase residual, transients and full-band quality would need assessment.

4. **Do not copy expensive internals or training parameterizations automatically.** Restoring high-rate multistage processing or periodic activation across many channels would work against the CPU target. Weight normalization can be removed for inference but changes optimizer parameterization during training; the current student's calibrated normalization and LayerNorm are not evidence of a missing required component. Keep these stable unless a specific failure supports an isolated change.

5. **No latent-cycle constraint by default.** The student already receives the identical raw latent tensor as the teacher. Re-encoding its 48 kHz output requires returning to the teacher encoder's 16 kHz interface, and even teacher reconstruction need not re-encode to identical original latents. Any such future objective needs a teacher self-consistency floor; it is not a missing identity guarantee today.

The safest next architecture experiment remains tanh alone. Preserve existing normalization, raw latents, context, frozen teacher and training objectives while testing it. Keep startup and phase-coupling differences as named, evidence-driven follow-ups rather than accumulating unproven changes.
