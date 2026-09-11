**Decoder recommendations after the diagnostic audit**

Research and planning review, 9 September 2026. This document proposes the next work. It does not change the student, teacher, optimizer, cached targets or training state.

The evidence supports retaining the current low-rate ConvNeXt decoder while addressing reproducibility, actual optimizer updates and amplitude control in that order. It does not establish an architectural capacity ceiling. Silence and loud peaks need separate treatment: the full parameter update reverses a locally favorable quiet direction, whereas peak error worsens even with a small fraction of that update.

Use the targeted step-8,490 checkpoint as the primary starting point because its natural quiet residual is lower than the complex candidate's. Preserve the complex checkpoint as the spectral-quality alternative and the step-8,090 parent as the historical control. This is a proposed starting point, not promotion of a production model.

**What the upstream implementations actually establish**

The pinned AudioVAE2 decoder ends with Snake, a causal seven-tap multichannel-to-waveform convolution and tanh. It uses weight-normalized causal convolutions, zero-left padding and sample-rate conditioning. The selected configuration disables noise injection. Its encoder returns posterior means. No dedicated inference silence gate is present. Tanh bounds amplitude but is almost the identity near zero, so it does not explain the low silence floor. [Pinned AudioVAE2 source](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py)

Our inspected SuperTonic 3 graph uses ten ConvNeXt blocks with GELU, a causal kernel-three head convolution, PReLU, a bias-free waveform projection and flattening. It has LayerNorm, fixed BatchNorm and replicated left padding. It contains no Snake, terminal tanh, output limiter, sample-domain overlap-add or silence gate. Its output frames have 512 samples; ours have 480 to meet the 48 kHz contract. The common structure already exists in our student. These are findings from the pinned graph, not an inference from a product speed claim. [Graph audit](../convnext-restart-plan/supertonic3-static-graph-audit.json), [pinned model artifact](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/onnx/vocoder.onnx)

The SuperTonic paper documents mel reconstruction, multi-period and magnitude-spectral discriminators, feature matching and AdamW. Its autoencoder trains jointly and its reported evaluation emphasizes perceptual quality and pitch. That does not establish exact waveform-phase recovery from our frozen encoder. The released V3 encoder configuration also differs from the paper, so the paper cannot be treated as a complete V3 training specification. [SuperTonic paper](https://arxiv.org/html/2503.23108v3#S3.SS1), [released configuration](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/tts.json)

An official collaborator disclosed an earlier AudioVAE recipe with DAC-style short multiscale mel, adversarial and feature-matching losses, plus cosine learning-rate decay and warmup. It is useful methodology, not a verified complete AudioVAE2 recipe. An unrelated V2 issue reply does not establish official training settings. [AudioVAE loss disclosure](https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3767009845), [schedule disclosure](https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3771286634)

Neither release demonstrates a special silence correction that we can simply copy. Our earlier statements about a missing SuperTonic Snake activation or a required sample-level output filter would therefore be incorrect.

**Recommended work, with a decision for every gap**

| Gap | Evidence and decision | Recommended next action |
| --- | --- | --- |
| G01: incomplete component panel | Closed. The corrected diagnostic includes all 285 crops. | Keep explicit source/category coverage assertions. Do not change the decoder for this measurement bug. |
| G02: output gradients used as a proxy for learning | Closed as a measurement gap. Native optimizer replays now expose actual behavior. | Retain actual parameter-update and post-update audio checks in the next small comparison. |
| G03: quiet gradient concentration | More quiet loss weight is not justified: the combined local direction already helps, and quiet gradients can dominate. | Keep current weighting initially. Assess actual quiet improvement after joint updates before increasing any silence penalty. |
| G04: loss-balancer drift | Complex FM+GAN averaged 40.20% against a 30% target; targeted averaged 31.95%. | Keep targeted's balancer unchanged in the first comparison. Later isolate faster EMA tracking or fixed calibrated coefficients in the complex branch, with measured contributions. |
| G05: losses stop rewarding the teacher | Not supported by the five tested interpolation points. | Retain teacher waveform, mel, GAN and feature matching. There is no evidence here for deleting GAN or replacing the whole objective. |
| G06: collapsed output basis | Ruled out: rank 480 and moderate condition numbers. Learnability remains open. | Keep head width. If stable training still plateaus, use a deliberately small training-only fit test before proposing more capacity. |
| G07: repeating quiet residual | A fixed 480-sample component explains only part of natural quiet error. Upstream features dominate the measured parent-to-candidate waveform changes; this does not identify the origin of every artifact. | Optimize the full existing student; distinguish periodic and varying residual. Do not subtract a template or add another smoothing filter as the presumed cure. |
| G08: all quiet windows fail | Real under the engineering checks; those checks have not been calibrated as audibility limits. | Track residual and output level against the teacher, including worst cases. After update stabilization, test controlled input-level coverage if a floor persists. Preserve whispers and breaths. |
| G09: rare loud peaks | Interior events occur in speech as well as laughter. Smaller updates alone still worsen peak error. | Separately adapt a teacher-style tanh head. Require transient fidelity as well as bounded output; the earlier short trial was not quality neutral. |
| G10: insufficient history | Not supported on eight natural sources: 30 latent frames match longer history to numerical precision. | Keep current context and padding. Do not spend extra inference work on more history for this issue. |
| G11: timing and gain | Case-dependent; one whistle has high cosine but substantial level/timing error. | Score raw gain, lag, waveform and spectral fidelity separately. Keep uncorrected audio as the primary result. Use stronger spectral supervision only for remaining demonstrated errors. |
| G12: validation coverage | Small language cohorts and missing reviewed crying/giggling/shouting groups remain. | Add source-disjoint, event-reviewed coverage and preserve an untouched final set. Do not train on the recordings used to diagnose failures. |
| G13: bundled earlier interventions | Earlier arms cannot isolate every historical effect. | Use one changed factor per continuation from the same parent and data schedule. Combine only retained improvements afterward. |
| G14: full updates reverse quiet benefit | Confirmed on the difficult update panel, not yet across the training distribution. | Compare the current rate with a smaller joint generator rate, preserving optimizer state and the rest of the recipe. |
| G15: parameter-family interaction | Joint Muon/AdamW displacement is much worse for quiet error than either subset on the tested step. This does not prove optimizer incompatibility. | Scale and assess the joint generator update first. Do not remove Muon, reset moments or freeze late blocks based on one batch. |
| G16: readout generalization | Extended ridge fits improve training MSE but trade unseen waveform/quiet quality for mel gains. | Do not install a fitted head or output bias as the solution. Retain the learned head and allow upstream features to adapt. |
| G17: encoder-cache reproduction | Two laughter sources have materially different fresh latents despite verified input and cache provenance. Cached latent/target pairs remain coherent. | Resolve or explicitly isolate the discrepancy before sustained training. Never silently overwrite the original cache or merge incompatible baselines. |

The numerical evidence and qualification limits for each row are in the [completed diagnostic report](report.md).

**First: establish a reproducible teacher baseline**

For the two discrepant laughter sources, compare the actual in-memory encoder parameters and buffers, input tensors, execution dtype, batch shape and runtime settings against the historical preparation path. A checkpoint-file hash alone does not prove the effective loaded module state. If the historical execution can be reproduced, locate the first layer where outputs diverge. Compare those cases with sources that already reproduce correctly.

Extend the check to a bounded source-stratified sample to estimate prevalence, rather than assuming two cases describe the entire corpus. Record a preparation identity that includes effective state and runtime information. The acceptance condition is reproducible encoding within a declared numerical tolerance, or a documented execution difference with separately versioned targets and metrics. Re-render only affected material if justified. The existing cached pairs are not established to be corrupted.

This is especially important for any volume augmentation: a source must be changed before the frozen encoder, then both its latent and teacher waveform must be prepared together. Multiplying existing latents or only changing the teacher waveform does not preserve the learned encoder-decoder relation.

**Second: make the existing optimizer improve quiet audio reliably**

The useful direction is present, but the full step can move too far along it. On targeted, quiet squared error improved 5.23% at one-quarter of the observed update and worsened 34.96% at the full update. Peak-excess energy still worsened 4.10% at one-quarter. This directly motivates a smaller-update comparison for silence, while showing why it is not a complete peak fix.

Start with just two continuations from targeted step 8,490: the current generator rate and one-quarter of that rate, applied to both generator optimizers. The latter is a candidate, not a validated optimum. Keep discriminator rate, loss and balancer policies, clipping, calibrated model-normalization statistics and matched training data unchanged. Start with identical optimizer and EMA states, then let moments, EMA estimates and discriminator weights evolve normally in each arm. The original path interpolation was not a multi-step learning-rate test, so improvement must be established by actual continuation across varied batches.

Use the existing short comparison scale, approximately 400 updates, as an initial decision budget. Check early finite-state health and evaluate quality at the end, rather than repeatedly running full quality suites every few steps. Preserve each branch. Extend a favorable branch only after it improves both natural and encoded-silence residual without materially worsening speech/expressive fidelity. Before execution, record numerical regression limits for aggregate waveform/mel scores and worst-case expressive/language behavior in the experiment configuration, using the existing criteria and evaluation variability. A pilot screening margin must not be presented as final perceptual equivalence. This is a progress gate, not a demand for final 0.99 reconstruction after a short run.

Do not simultaneously reset momentum, raise clipping thresholds, remove Muon or freeze blocks 8 and 9. All were implicated only through current-state behavior; none was individually proven defective. AdamW uses stored moments and decoupled decay, and Muon transforms its update, so a gradient norm or clipping frequency alone is not an effective-step measurement. [AdamW implementation contract](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html), [Muon implementation contract](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Muon.html)

The complex branch's loss controller should be a separate later comparison. Its observed gradient growth outpaced its EMA estimate. A shorter memory could help but also increase batch noise; a fixed coefficient could simplify the controller but allow contributions to drift as the discriminator learns. Neither is established as better. Select on actual updates and audio, not on achieving a nominal percentage alone.

**Third: adapt the output bound without hiding damaged peaks**

The teacher's tanh is the strongest structural candidate for guaranteed amplitude range. Its near-zero behavior cannot remove our quiet residual. This is a distinct peak experiment, after the update-size result, using the trained student rather than random initialization.

The prior 200-update tanh trial removed overshoot, but natural MAE rose 0.50%, quiet RMS rose 1.68%, and the largest output peak was 0.922 against a teacher maximum near 0.995. Therefore, repeat only with a better-defined adaptation question and the stabilized training control. A bounded waveform with suppressed transients does not pass.

Keep the same paired waveform and perceptual objectives initially. Record the teacher's pre-tanh waveform and the student's pre-tanh output on matched loud events to distinguish insufficient logit adaptation from errors already present before bounding. This is a diagnostic use of an existing teacher activation, not a requirement to match hidden channels or add another loss immediately. Preserve full-student adaptation; the readout probes do not justify a permanently head-only solution.

Acceptance requires every output to be finite and within [-1, 1], no replacement of overshoot with persistent saturation, and preserved teacher-relative transient shape, loudness, crest factor and quiet behavior. Review Sindhi speech, laughter and other loud events, not just aggregate maximum amplitude. A pointwise tanh changes no causal history, but its final exported one-thread CPU cost and streaming parity still require validation on Intel, AMD and Apple before deployment.

**Fourth: address any remaining silence floor through matched examples**

Targeted data already improved natural quiet RMS by 11.4% relative to the original checkpoint and by 29.0% relative to regular continuation. Retain that mixture. It includes roughly 5.08% explicitly labelled expressive/whisper sources, 2.17% natural quiet and 2.90% transitions, so the next proposal is not simply adding the same categories again. [Corrected comparison](../convnext-corrected-experiments/report.md), [sealed data selection](../convnext-corrected-experiments/data-selection.md)

If stable updates still leave the measured output floor, make the next isolated data change controlled level coverage: distinct training-source crops at natural and low input levels, plus genuine encoded-zero and quiet-transition examples with matched teacher targets. Draw natural low-level speech, breath, whisper, whistles and expressive tails; synthetic silence alone does not represent the varying natural residual. Audit actual event spans because a recording labelled laughter can contain mostly silence or another sound.

Use the existing raw waveform reconstruction first. Do not increase quiet weights simply because the pass count is poor, restore inverse-RMS amplification, or force teacher-quiet intervals to literal zeros. A noise gate could erase valid breaths. We should learn the teacher's small residual and meaningful low-level detail. The zero-input RMS is about 9.63e-6 for the teacher, versus 1.43e-4 and 1.23e-4 for targeted and complex. That is a measurable floor gap, not proof every failure is audible.

**Fifth: improve remaining phase and transient fidelity with training-only methods**

The corrected complex discriminator already improved mel error 2.95% and high-frequency magnitude error 5.99% relative to targeted, while quiet RMS worsened 1.79%. Keep it available, but do not declare it the universal winner until its achieved weighting and update behavior are comparable.

DAC's spectral discriminator uses real and imaginary components and separate bands. Its preprocessing also removes DC and peak-normalizes each waveform. Preserve our absolute-amplitude teacher comparison instead of copying that preprocessing. Complex discrimination provides phase-sensitive evidence, not a guarantee of exact sample phase. [DAC discriminator](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/discriminator.py)

Shorter mel analysis remains a conditional transient experiment. The old trial added only 256/512-sample windows, altered the scale mixture and inherited stale loss statistics. It did not test DAC's full 32-through-2048 schedule. The official DAC work motivates short-time coverage for transients, but those exact scales and weights are not automatically optimal at our sample rate. Recalibrate a changed combined mel branch, retain longer windows, and evaluate raw waveform, quiet and transient fidelity together. These analyses add training cost and no decoder inference operations. [DAC paper](https://arxiv.org/html/2306.06546v2), [earlier short-window trial](../convnext-fusion-experiments/report.md)

Keep 0.99 nonquiet waveform correlation as a final reconstruction goal alongside level, phase, quiet residual and listening. A complex whistle already had cosine 0.99276 despite a substantial gain/timing error. Do not fit away those differences in the primary score. For the same reason, neither upstream perceptual scores nor their training duration guarantee our stricter reconstruction target.

**What to leave alone unless new evidence changes the diagnosis**

- The frozen 64-channel AudioVAE2 encoder and identical teacher/student latent tensors. Decoder-only training does not need a new encoder KL objective or latent-cycle requirement.
- Current causal context, replicated padding and calibrated model normalization. Loss-balancer calibration is a separate issue from model-normalization calibration.
- The ten-block low-rate body and learned full-rank projection. The seven-tap one-channel postfilter worsened quiet RMS by 10.52%; it is also not equivalent to the teacher's multichannel synthesis layer.
- Source-separated evaluation and raw amplitude. Expand sparse languages and reviewed nonverbal groups, with a final unseen panel to limit overfitting to these repeatedly inspected development crops.

If the stabilized, properly paired student still cannot fit selected training examples, investigate learnability with a small debugging fit before another architecture change. Such a test may reuse training examples under the user's debugging allowance; it must not contaminate final evaluation. Failure of one optimization setup alone would still not prove a capacity ceiling.

All of the proposed changes except the optional output activation concern preparation, optimization, training supervision or evaluation. The intended cheap causal decoder structure remains intact. This review contains no new quality or RTF measurements.

**Evidence and provenance**

The main numerical sources are the [diagnostic audit](report.md), [initial six-branch screen](../convnext-fusion-experiments/report.md) and [corrected four-branch screen](../convnext-corrected-experiments/report.md). Different screens have different declared panels; percentages above always refer to their own matched comparison.

The SuperTonic graph interpretation uses revision `3cadd1ee6394adea1bd021217a0e650ede09a323`, SHA-256 `085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba`. The official repository now redirects to an archive; the graph bytes at the new archive destination were not independently refreshed in this review. This does not change the identity of the inspected local artifact. [Official repository](https://github.com/supertone-oss-archive/supertonic)
