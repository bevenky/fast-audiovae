# Teacher spectral supervision and head adaptation

Run a bounded four-arm comparison from the unchanged step 8,890 checkpoint. Keep AudioVAE2's frozen encoder and exact 64-channel latent inputs, along with the student's causal ConvNeXt body and direct waveform synthesis. Do not add layers, change PReLU, clip peaks, or install an inference silence detector in this experiment.

| Arm | Parameters updated | Objective |
| --- | --- | --- |
| Joint waveform | Existing head convolution, PReLU and output projection | Prior selective waveform objective |
| Joint spectral | Same joint head | Same waveform objective plus teacher mel |
| Projection waveform | Final output projection only | Prior selective waveform objective |
| Projection spectral | Final output projection only | Same waveform objective plus teacher mel |

The waveform objective remains the preceding experiment's teacher quiet MSE plus nonquiet original-student preservation penalty. The fixed training-pool quiet and nonquiet baseline errors normalize these branches; the preservation multiplier remains 100. The original student's nonquiet waveform is a temporary safeguard, not the ultimate teacher target. New spectral supervision compares against AudioVAE2 across all eligible valid audio.

Use the same deterministic 2,048-source fitting split and source/audio-hash-disjoint 256-source selection split. Both exclude the canonical panel. Each arm uses 256 updates of eight distinct recordings, with no repeated fitting recording within an arm. The same ordered data are reused across arms for the controlled debug comparison. Cache frozen pre-head features in RAM and verify exact singleton replay before updates.

Reuse the existing three-resolution magnitude mel definition: Hann windows of 1,024, 2,048 and 4,096 samples at 48 kHz, quarter-window hops, Slaney filters and the existing log floor. Pool time-frequency counts per resolution, then average resolutions. Evaluate complete contiguous valid waveform slices without zeroing quiet or active masks or joining separate intervals. Valid slices shorter than 4,096 samples retain waveform supervision; explicitly count any omitted spectral exposure.

Set one common teacher mel coefficient from the first four fitting batches, before parameter updates. Compute gradients with respect to the original predicted waveform, not parameter gradients that depend on the trainable scope. The coefficient is the square root of summed waveform-gradient energy divided by summed mel-gradient energy. This sets equal initial output-gradient energy over those training batches and stays fixed in every arm. Record per-batch norms and alignment. The preservation gradient is zero initially; this calibration does not claim a constant loss balance during training.

Use fresh matched AdamW states, no weight decay, FP32 and a global gradient norm cap of 1. Start the common learning rate at 0.000001. A training-only first-batch trial may halve it at most six times until every arm satisfies the same quiet, nonquiet-displacement and per-source mel safeguards. Restore all trial state before training. Equal rates do not guarantee equal acoustic displacement; record actual displacements. Stop if no rate qualifies rather than weaken the criteria.

Score the separate selection split at steps 64, 128, 192 and 256. Strengthen the prior source-level checks: every recording's mel error must stay within 1% of its baseline plus a 0.000001 numerical tolerance, alongside the existing aggregate waveform, nonquiet preservation and peak checks. A qualified silence candidate also needs at least 10% lower natural quiet residual RMS. Select only qualified checkpoints; report final results even if none qualifies.

Evaluate fixed candidates on all 285 canonical development crops using the sealed teacher targets, masks and sample counts. Keep all prior aggregate and group reconstruction checks, and add per-source mel checks rather than accepting an average that hides a large regression. Report quiet, active and transition spectral errors by classifying each FFT frame from the teacher-only quiet mask over its full support, after the transform. A frame is quiet if all its samples are quiet, active if none are quiet, and a transition frame otherwise. Report absent support explicitly. These are diagnostic categories, not new inference branches.

No candidate is promoted automatically. Original engine state, normalization, checkpoint and caches must remain unchanged. Only a candidate passing quality proceeds to CPU-only one-thread streaming parity and timing. Preserve the inference graph in all four arms, and do not label GPU experiment time as CPU RTF. Training uses the existing H100 pod and writes separate small artifacts to temporary storage; the production training run stays paused.
