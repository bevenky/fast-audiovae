# AudioVAE2 architecture re-audit and late-update replay

The original AudioVAE2 source and the pruning implementation were rechecked, and the requested 4,500-to-5,000 replay completed. No missing released decoder component, incorrect channel connection or corrupted teacher target was found. The replay shows substantial, repeated output-gain oscillation, which makes training-update stability a stronger next target than further architecture changes. It did not meet the predefined exact-replay requirements, so its trajectory cannot be presented as an exact reconstruction of the original run.

## Original architecture and training evidence

The current official decoder source is byte-identical to our pinned source. The student preserves the original causal geometry, Snake activations, depthwise convolutions, pointwise projections, conditioning, weight normalization, final convolution and tanh. All nine residual units within stages 2–4 remain present, as do the original units outside the group. There is no released silence-specific gate or DC corrector missing from our copy. [Official pinned decoder](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py).

This candidate narrows stage-2 output channels from 512 to 256 and stage-3 output channels from 256 to 128. The complete group's input and output remain unchanged. Narrowing still removes learned contributions to later convolutions, including cancellations. Correctly connected retained layers do not automatically compute the same function. Joint distillation must recover that function with less internal capacity. [Detailed pruning analysis](pruning-reaudit.md).

The public repository does not supply a complete V2 codec-pretraining trainer. Its TTS fine-tuning optimizer is not a VAE recipe. Authenticated older AudioVAE disclosures document a much larger effective batch and warmup/cosine decay, but those values cannot simply be declared mandatory V2 compression settings. KL cannot update this decoder through a frozen encoder; adding GAN losses is not an established fix for the measured amplitude drift. [Source and recipe audit](upstream-reaudit.md).

## What the replay checked

The isolated replay restored the original step-4,500 group weights, AdamW state and RNG, and used the exact next 1,500 cached sources, in the same order. It performed 500 updates with the unchanged three-source accumulation, learning rate and loss coefficients. Sixteen focused tests passed locally and remotely before launch. Original files and frozen modules were preserved. The replay stopped at 5,000 and did not extend the main run.

All 1,500 cached target waveforms matched the teacher's scored output **bitwise**, including the original causal context and valid tail handling. This interval provides no evidence of conflicting waveform and hidden-feature targets caused by stale or corrupt caches.

The strict reproduction checks failed. The first discrepancy was at step 4,501: waveform loss differed by approximately 2.33e-10 and mel loss by approximately 2.98e-8, while the feature loss matched. After 500 updates, effective-weight relative L2 difference was approximately 0.0121%; the four final waveform MAEs differed by 0.18–0.48% from the original. All four exceeded the existing endpoint tolerance. No threshold was relaxed, and the replay model was not selected as a candidate. [Replay report](replay4500-5000/completed.json), [numerical discrepancy measurements](replay4500-5000/discrepancy-summary.json).

## What its trajectory shows

| Fixed recording | Sampled student/teacher RMS-gain range | Median gain | Direction reversals |
|---|---:|---:|---:|
| Latin American Spanish | 0.814–1.121 | 0.991 | 10 |
| Kannada | 0.886–1.133 | 0.984 | 9 |
| Kashmiri | 0.895–1.144 | 0.992 | 9 |
| Whistling | 0.641–1.064 | 0.871 | 14 |

These are 21 observations at 25-update intervals, not extrema over every update. Gain 1 means matching teacher RMS; it does not imply waveform fidelity.

The three speech cases repeatedly become quieter and louder, with partly shared motion. Pairwise correlations of their gain changes range from 0.279 to 0.773, so this is not one identical global volume change. Their mean gain drops approximately 0.1036 between 4,900 and 4,925, then rises approximately 0.1063 over the final 25 updates. All three reach their sampled maximum at 5,000. This is oscillation rather than a steady irreversible increase in volume in the replay. Quiet-window failures remain largely unchanged, so the active-volume oscillation is a separate issue from the stationary tiny offsets and startup/transient errors. [Trajectory audit](replay4500-5000/replay-trajectory-audit.md), [execution audit](replay4500-5000/execution-audit.md).

On 33 of 500 updates, the actual AdamW displacement has a positive dot product with that update's weighted minibatch gradient. The final 25-update interval contains six such steps, versus a median of one per interval. This is a local directional measurement, not proof that the finite minibatch loss increased or that AdamW is implemented incorrectly. No obvious discrete dataset transition coincides with the final rise. The data mixture remains varied throughout.

## Interpretation and next action

The evidence does not support removing another layer, adding an omitted AudioVAE component or broadly reducing mel/feature losses. The earlier fixed-case gradient audit showed that waveform gradients dominate those losses on the inspected failures. The narrowed group still changes its shared output enough that the intact suffix produces substantial volume changes.

The leading hypothesis is insufficiently stable optimization of a shared gain-sensitive direction. A three-source effective batch, constant learning rate and optimizer history are plausible contributors. The replay's oscillation supports investigating that mechanism, but does not uniquely prove which setting causes it or establish the exact original trajectory. Capacity limits and expressive-data coverage remain separate questions.

The next controlled comparison should change one stability factor: increase effective batch by accumulating more singleton examples, preserving the verified singleton teacher and student paths. Compare against the current accumulation at equal audio/source exposure, using the same starting checkpoint and all fixed regional checks. At equal source exposure, this also reduces optimizer-update count and changes Adam's history measured in examples. It tests the practical batching policy, not gradient variance alone. Judge sustained gain stability and waveform fidelity across evaluation points, not the most favorable checkpoint. This comparison has not been started. A decay schedule is a separate possible comparison, with precedent in older disclosed recipes; it should not be changed at the same time if attribution is the goal.

Once a correction is demonstrated, restart the intended continuation from the preserved 1,000-step anchor as requested. There is currently no demonstrated training-code bug or validated fix that justifies calling a restart repaired. Main training remains paused, no additional layers were removed, and no model was committed or promoted.
