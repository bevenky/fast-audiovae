# Layer observations on actual quiet and loud recordings

Both frozen decoders were traced on four fixed recordings: two Mandarin quiet crops selected by teacher quiet coverage, the previously identified Sindhi peak crop and the laughter peak crop. This is a diagnostic case study, not a representative quality benchmark.

Every case used its original context and identical 64-channel latents for teacher and student. Teacher hooks recorded 119 points, including conditioning, activations, upsampling, residual blocks and the final head. Student hooks recorded 67 points across its adapter, all ten ConvNeXt blocks and output head. These are observation counts, not counts of independently trainable layers.

All hooked teacher outputs exactly equal ordinary execution. Their scored samples exactly equal the frozen canonical targets. All student outputs with hooks and gradients enabled exactly equal ordinary execution. Every observed tensor is finite. The original model and optimizer state hashes are unchanged. No gain, parameter or checkpoint was modified.

## The teacher's bounded output is doing substantial work on peaks

| Existing failure case | Teacher before tanh, max absolute | Teacher after tanh, max absolute | Student max absolute | Student samples above full scale |
|---|---:|---:|---:|---:|
| Sindhi speech | 2.96334 | 0.99468 | 1.29663 | 196 |
| Laughter | 2.97442 | 0.99480 | 1.19931 | 58 |

These are maxima over each scored crop, not necessarily the same sample positions across models. The teacher's tanh is monotone, so its pre/post maxima directly correspond. Before tanh, 2,041 teacher samples exceed full scale in the Sindhi crop, and 294 in the laughter crop. After tanh, none do.

The teacher has learned a synthesis path that uses its final nonlinear compression. Our student is learning the teacher's already-compressed waveform with an unbounded head. The student therefore lacks the same mathematical output bound. This does not show that its hidden values should match the teacher's hidden values, or that adding tanh to the existing trained student would preserve its learned waveform. A student output of 0.9, for example, would become about 0.716 if tanh were simply appended.

For the two quiet crops, tanh changes the teacher's scored waveform by RMS 5.06e-10 and 2.99e-8. For stationary digital silence, the change is exactly zero in FP32. Thus the teacher's peak protection and its silence reconstruction come from different mechanisms.

## The last block has an identifiable tradeoff

For each student residual block, the diagnostic computes the derivative of the selected output error with respect to an infinitesimal gain on that block's existing residual contribution. The actual gain remains one. This uses the real downstream network; it neither removes a block nor substitutes teacher features.

Block numbers below are one-based. The internal code names them `blocks.0` through `blocks.9`.

| Observation | Measured local direction |
|---|---|
| Sindhi peak-excess error | Only block 10 has a positive gain derivative; blocks 1–9 have negative derivatives |
| Laughter peak-excess error | The same pattern: block 10 positive, blocks 1–9 negative |
| Ordinary nonquiet, nonovershooting samples in those two crops | Block 10 has a negative reconstruction-error derivative in both |
| Teacher-defined quiet samples in both Mandarin crops and the laughter crop | Block 10 has a negative reconstruction-error derivative in all three |

A positive derivative means an infinitesimal reduction in that block's residual contribution would reduce that error. A negative derivative means the same reduction would increase it. Therefore weakening block 10 would locally trade lower peaks for worse ordinary reconstruction and quiet fidelity in these examples. This argues against treating it as a disposable or universally defective layer.

Blocks 7 and 8 have positive quiet-error derivatives in both Mandarin cases. However, their ordinary-speech control derivatives reverse sign between the two crops. That likewise does not identify a uniformly beneficial gain adjustment.

The quiet crops contain only 60 ms and 200 ms of nonquiet control samples, respectively. Their controls are limited. The independent loud-crop controls include much more audio and exclude both teacher-defined quiet samples and existing student overshoots.

## What is now localized, and what is still open

The combined evidence identifies three distinct issues:

1. The stationary silence pattern reaches the waveform through the student's frame-to-480-sample readout. The teacher's final multichannel convolution nearly cancels its periodic component.
2. On the two loud examples, the student's last residual block has a local peak-versus-reconstruction tradeoff. The teacher's final tanh supplies an explicit bound that the student does not have.
3. The separate optimizer comparison establishes that the realized update magnitude can reverse an otherwise helpful quiet-error direction.

This provides specific locations and mechanisms for designing a controlled intervention. It does not establish one broken layer that explains every failure. It also does not show that a finite block-gain change, optimizer reset or appended tanh is a safe fix.

The next design decision should preserve the useful last-block representation, address output bounding as a distinct requirement, and evaluate silence against the actual teacher response. If a bounded head is tested, its migration must account for the existing student already predicting post-tanh targets; blindly appending the teacher's activation changes those predictions. Continued training or a revised head remains unstarted pending that decision.

[Natural trace and derivatives](natural-layers.json) · [Stationary paired trace](paired-layer-results.md) · [Optimizer evidence](optimizer-results.md)
