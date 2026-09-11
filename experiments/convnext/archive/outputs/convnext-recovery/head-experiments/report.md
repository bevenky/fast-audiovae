# Decoder head experiment results

Keep the existing step 8,890 checkpoint. None of the learned head candidates passed the predeclared quality screens or solved stationary silence, natural quiet and peak overshoot together. The results reject these specific repairs; they do not prove that the ConvNeXt backbone must change.

## What was tested

All ten ConvNeXt blocks, normalization and the existing nonlinear head features remained frozen. Only the final 480 × 2,048 readout weights changed. These fits add no layer, state or lookahead. The teacher, encoder and original checkpoint remained unchanged.

The fit used 2,048 distinct recordings and 500,347 complete synthesis frames, about 83.4 minutes of scored audio. Another 256 source-disjoint recordings selected regularization. Both sets excluded the development panel by source ID and audio hash. The selection includes multilingual speech and material from Freesound, CREMA-D and Thorsten Emotional. This was a bounded calibration experiment, not a full training restart.

## Results on the existing development panel

Lower error is better. Changes below compare with the unchanged checkpoint on the same 285 crops from 147 sources. The natural subset has 282 crops from 144 sources.

| Candidate | Steady-silence error | Natural quiet error | Waveform MAE | High-frequency magnitude error | Maximum peak | Decision |
|---|---:|---:|---:|---:|---:|---|
| Existing decoder | Reference | Reference | Reference | Reference | 1.297 | Retain |
| Shared silence-pattern fit | −88.43% | +2.20% | +2.16% | +5.74% | 1.313 | Reject |
| Four-phase silence fit | −99.98% | +2.42% | +3.27% | +11.45% | 1.306 | Reject |
| Existing decoder with clipping control | Unchanged | Unchanged | −0.017% | Essentially unchanged | 1.000 | Range protection only |

Silence and quiet columns use residual RMS against the teacher. Baseline steady-silence residual is 4.3793e-5; natural quiet residual is 2.58495e-4. Matching all four phases reduces the former to 7.3423e-9, while increasing the latter to 2.64760e-4. These are different signal populations, not contradictory measurements.

Clipping removed all 588 overshoot sample observations, which occur across eight crops from six sources. These are observations in possibly overlapping crops, not 588 distinct events. It passed the measured reconstruction screen but did not improve quiet audio or teach the network better transient reconstruction. Adding clipping to either fitted silence head did not rescue its quality failures. No clipping change was installed in production.

## Real-audio readout fits

Both broader fitting families failed on the separate 256-recording training-validation split, before development-panel scoring. Four fixed regularization strengths were tested for each family.

| Fit family, strongest regularization | Waveform MSE change | Waveform MAE change | Natural quiet RMS change | Result |
|---|---:|---:|---:|---|
| Ordinary teacher waveform target | −1.63% | +0.22% | +0.56% | No quiet improvement; fails the fixed quiet gate |
| Actual teacher pre-tanh target, followed by tanh | +7.78% | +13.28% | +4.94% | Reject this linear migration |

The ordinary-target fit narrowly missed the 1% quiet-MSE limit: its quiet MSE rose 1.113%. Relaxing that limit would still not produce the intended substantial quiet improvement. The pre-tanh fit failed materially. A linear readout fit is not equivalent to jointly adapting a nonlinear synthesis head.

## Why this is informative

The existing features can represent the teacher's stationary silence response. However, the tested fixed readout corrections also change useful real-audio predictions. A near-perfect synthetic silence score therefore does not justify keeping them.

The teacher learns its final features and synthesis weights together. These experiments held those features fixed. The next justified design question is whether joint adaptation of the student's existing causal head convolution, PReLU and output projection can preserve useful features while learning cancellation and bounded amplitudes. Initially keep the ten ConvNeXt blocks fixed. This uses existing computation; a selected tanh would add only its pointwise operation.

That next experiment should use actual post-tanh teacher reconstruction quality alongside real quiet trajectories, with stationary silence monitored rather than imposed as an absolute single-fixture equality. It is a proposed test, not a demonstrated fix, and has not started. These results do not justify adding a silence gate, scalar output filter or weakening the final ConvNeXt block.

## Teacher and state verification

All 2,048 captures reproduced the entire saved latent crop, teacher waveform crop, and original whole-source cache key exactly. Hooked and ordinary teacher outputs also matched. No target or tolerance was changed. The initial diagnostic found startup-dependent encoder rounding and a cropped-versus-whole-source execution difference; explicit warmup and the original full-source geometry resolved the exact replay checks.

The completed GPU calibration and quality run took 258.8 seconds after setup. This is experiment time, not a CPU decoding RTF. The original engine, optimizer state, teacher, caches and checkpoint were verified unchanged. No candidate was promoted, committed or merged.

## CPU streaming checks

On the Runpod AMD EPYC 9654, using one CPU thread with CUDA disabled, all 36 cases emitted exactly the expected sample counts and preserved empty-call/state behavior. The original decoder and clipping control passed every tested 80/160 ms case. One already-rejected shared-pattern head failed the unchanged 2e-6 maximum batch/stream difference limit: 2.14577e-6 on the 80 ms laughter case. The other 35 checks passed. The threshold was not relaxed.

The following are median RTFs for the experimental PyTorch student with unchanged baseline weights, two warmups and five interleaved measurements. They isolate output-activation cost. They are not optimized ONNX deployment measurements or a comparison with Mimi.

| Chunk | Original output | Clipping control | Tanh cost probe |
|---|---:|---:|---:|
| 80 ms | 0.2208 | 0.2212 | 0.2214 |
| 160 ms | 0.1133 | 0.1170 | 0.1119 |

These small timing differences do not establish a speedup or a consistent activation penalty. Tanh timing here is a cost probe, not a qualified tanh checkpoint. Every runtime tensor stayed on CPU, no teacher or encoder was invoked, and the original engine state and RNG were restored.

## Limits

This is the existing development validation panel, which has been inspected in earlier rounds, not a fresh final test. The six-second silence fixture belongs to the same signal family as the eight-second calibration recording. Its improvement verifies the constraint, not natural-audio generalization. Reconstruction screens do not establish perceptual equivalence. Hard anchoring, fixed regularization and sample-pooled squared error also limit what failed readout fits can establish about model capacity.

## Evidence

- [Fixed plan](plan.md) and [layer-to-experiment reasoning](mechanisms.md).
- [Candidate comparisons](results/comparisons.json), [ordinary-target selection](results/post-selection.json), and [pre-tanh selection](results/pre-selection.json).
- [Completed experiment and state preservation](results/complete.json), [all teacher capture receipts](results/fit-accounting.json), and [startup replay diagnostic](source-repeat-diagnostic.json).
- [CPU streaming and timing results](cpu-heads.json).
