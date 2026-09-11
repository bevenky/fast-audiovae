# Minimal bounded-head experiment

This proposal starts from preserved step 8,890. It does not require changing any ConvNeXt block, the latent adapter, normalization or the teacher.

**Execution decision:** the hard clamp below is a frozen diagnostic control only. The learned architecture candidate is the faithful teacher output-parameterization migration: capture the actual teacher pre-tanh waveform, fit the student's existing final readout to that target, and apply tanh. Fit regularization is chosen using separate training-validation audio scored after tanh, with the quiet reconstruction guard. The baseline target-space readout fit uses the same data and method. The earlier optional optimizer fine-tune described below is not part of this experiment.

## What the evidence supports

The teacher's final tanh is used strongly on the two loud cases: pre-activation peaks of 2.963 and 2.974 become approximately 0.995. The student instead learned the teacher's already compressed waveform through an unbounded linear projection. Its peaks are 1.297 and 1.199. The teacher's peak protection and its silence behavior are different mechanisms: on stationary silence tanh changes exactly nothing in FP32.

The prior 200-update tanh candidate did enforce a bound, but after corrected rescoring its natural waveform error was 0.49% worse and expressive correlation declined. Appending tanh to the learned mapping changes every nonzero amplitude; 0.9 becomes approximately 0.716. This is a migration cost, not evidence that the teacher's activation is wrong.

The tenth block's local contribution helps ordinary reconstruction and quiet fidelity while increasing peak excess. Leave its representation intact in this experiment.

Primary source: [AudioVAE2 decoder implementation](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/modules/audiovae/audio_vae_v2.py). The local measured evidence is in the paired and natural layer reports, not inferred from the source alone.

## Frozen control: an exact bound with an unchanged interior

Use `audio = clamp(raw_audio, -1, 1)` after the existing final projection and flattening. This is two pointwise min/max operations, zero parameters, zero history, no lookahead, no frame-rate change. It can fuse into the existing output store. Its actual CPU cost remains to be measured; there is no claimed RTF result yet.

For every finite teacher target `t` in `[-1,1]`, the projection obeys:

`abs(clamp(x, -1, 1) - t) <= abs(x - t)`.

The same holds for squared error. Samples already inside the interval remain bitwise identical, including quiet samples. Therefore a frozen forward-only experiment cleanly separates enforcing a legal range from perturbing a learned interior mapping. Verify teacher bounds and finite values before relying on this property.

This is not a perceptual guarantee. A hard bound can introduce corners and high-frequency energy on the altered runs. Mel error, derivative errors and listening may worsen even when samplewise waveform error improves. It also says nothing about the quality of a retrained model after its weights change.

Run this inexpensive frozen comparison first across the complete existing canonical panel. Compare the original and clipped outputs on the exact same decoded arrays. No data-dependent threshold exists: the endpoints are the teacher's architectural range. Record changed samples/runs and maximum amplitude, MAE/MSE, unchanged-interior checks, existing mel and high-frequency errors, first/second waveform difference error against the teacher, per-source results, and the existing quiet windows. Scores must use floating point waveforms so file encoding does not silently clip the control.

## How to train it without dead gradients or a straight-through estimator

If the frozen screen is acceptable, keep the current waveform reconstruction term on `raw_audio`, with its existing scaling and balancing. Feed the bounded output to mel, adversarial and feature-matching objectives and to the discriminator's generated input. No extra weight or new target is needed. The existing raw waveform gradient still drives overshooting values toward the bounded teacher target.

For the MAE term this has a useful exact interpretation:

`abs(x-t) = abs(clamp(x)-t) + abs(x-clamp(x))` for bounded `t`.

Thus the existing raw MAE is exactly actual-output reconstruction plus a same-weight range penalty. It is not a fabricated gradient through clipping. Do not use straight-through gradients. Log raw and bounded errors separately so zero emitted overshoots do not conceal internal overshoots.

Preserve the checkpoint and all optimizer states. Both the bounded arm and a matched unbounded continuation must use the same declared learning-rate policy, distinct training sources in identical order, and the same random/discriminator views. The recent smaller-update evidence should inform that shared policy, not become a second varying factor. Do not reset one arm's optimizer or its loss-balancer history silently. An arm-specific change to the loss balancer itself would require an explicit protocol.

A finite experimental continuation is only warranted after the frozen screen; do not interpret a guaranteed output range as proof that training reduced internal peak error.

## Learned candidate: the teacher's output parameterization

Use the teacher's smooth final tanh, but train the student readout to predict the teacher's true pre-tanh signal first. Obtain that signal with a frozen hook before the teacher's existing final tanh, under the same corrected latents and full source context. Do not numerically invert a rounded or saturated waveform with `atanh`: the inverse is ill-conditioned near full scale and exact values of one have no finite inverse.

Freeze the existing latent adapter, all ten blocks, normalization and the first head projection/PReLU for a head-only feasibility fit. Train only the existing final 480-channel projection toward the true pre-tanh signal, with a matched post-tanh-target head-only control and identical data. Score emitted `tanh(raw)` against the ordinary teacher output. This tests whether the current hidden features can support the teacher's output parameterization without paying for new inference layers. It does not guarantee a fixed linear head can represent the needed nonlinear remapping.

This migration changes amplitude sensitivity across the entire waveform. The prior tanh candidate already demonstrated that simply enabling it and taking 200 updates can leave a quality penalty. The head-only pre-tanh feasibility fit explicitly addresses that migration before touching the backbone. The added CPU operation is one tanh per output sample; benchmark that cost separately from the min/max control.

`capture_teacher_pre_tanh(teacher, crop, quiet_config)` in `bounded_head.py` provides a scoped final-convolution hook. It runs the singleton decoder twice on the original full latent context, verifies native versus hooked output, verifies `tanh(captured_pre)` against native output over the complete tensor, and verifies native output against the canonical target under the existing scored mask. It retains the six-sample context exclusion, never invokes the encoder or inverse tanh, removes hooks on failures, and preserves teacher modes, parameter/buffer versions and existing gradients. The complete experiment separately verifies the teacher state hash.

## Excluded shortcuts

- A smooth bounded function cannot be exactly identity over the whole closed interval `[-1,1]` and smoothly flatten outside it. A smooth knee must change some valid near-full-scale samples. Choosing its knee from the failure panel would add tuning, so do not introduce one now.
- `tanh(atanh(clamp(x)))` merely implements clipping with extra transcendentals and numerical hazards. It is not a useful new representation.
- A hard clip applied only at the WAV writer is not a trained bounded decoder and would invalidate raw control comparisons.
- Removing or weakening block 10 would discard useful reconstruction behavior; the causal sensitivity signs specifically argue against it.
- Neither candidate is expected to fix stationary or natural quiet residuals by itself. Those need their own unchanged-interior comparisons.
