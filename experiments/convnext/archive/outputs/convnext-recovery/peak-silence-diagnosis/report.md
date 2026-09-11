# Peak and silence diagnosis at step 8,890

The audit found a concrete failure mechanism in a reproducible training update: the current batch's raw parameter gradient would reduce peak excess, but the native Muon/AdamW displacement increases it. Quiet audio has an additional finite-step problem. This supports investigating optimizer behavior before adding a peak penalty or changing the decoder.

This is a demonstrated mechanism for the tested state and batch, not proof that every bad training update has the same cause, or that a different optimizer will produce better final audio. No fix has been applied.

## What was inspected

- The preserved quarter-rate student at step 8,890, checkpoint SHA256 `f7a91a2f3bef6fdec2a37643dc661a97b18a24b9e25b822df49c1c4017ddc948`, and the step 8,490 baseline.
- Frozen forward passes on all 285 canonical crops, including 144 natural recordings and three fixtures. Teacher targets use the corrected, sealed execution contract.
- Direct waveform-gradient and shared-parameter-gradient probes on ten selected cases: six peak recordings, three quiet-regression recordings, and encoded digital silence.
- One 32-example native training update on a disposable copy, using saved moments, the actual discriminator update, original loss balancing, and the existing quarter rate. The identical update was replayed three times to retain increasingly detailed attribution. All endpoint metrics, gradients, updates and discriminator views matched exactly. These are not three independent batches.
- A CPU-only inventory of all 12,800 actual scored training input crops.

The retained model, optimizer state, normalization buffers and checkpoints are unchanged. Long training remains paused. Only scripts and diagnostic reports were created. No new quality-score suite or RTF benchmark was run.

## Peaks are short amplitude errors, not a block-join pattern

The canonical panel reproduces the previous maximum amplitude of 1.296632 and 588 above-full-scale sample observations. After removing overlapping crop observations, there are **384 distinct above-full-scale samples in 57 contiguous events across six recordings**. The baseline had 282 distinct samples; 115 appeared and 13 disappeared during the 400-update continuation.

Each event lasts between 0.0208 and 0.3125 ms. None of the 57 event peaks is within six samples of a 480-sample waveform-block boundary. The closest is 14 samples away. Some events occur early, but others occur more than two seconds into a recording, so startup handling cannot explain them all.

| Recording | Largest student peak, rounded | Teacher maximum within 5 ms of that peak | Best local lag within ±1 ms |
|---|---:|---:|---:|
| Sindhi | 1.2966 | 0.9939 | 0 samples |
| Kannada | 1.2062 | 0.9932 | 0 samples |
| Laughter | 1.1993 | 0.9933 | 0 samples |
| Punjabi | 1.0925 | 0.9930 | 0 samples |
| Spanish | 1.0465 | 0.9310 | 0 samples |
| Konkani | 1.0367 | 0.9652 | 0 samples |

At the largest event in each recording, timing adjustment does not improve local cosine. Across all events, 44 of 57 favor zero lag; this does not rule out smaller shape/timing errors in the remaining events. Overall crop RMS is below the teacher in every affected crop, so blanket attenuation would suppress audio that is already too quiet on average.

## The existing losses do request a correction

On the six overshooting examples, the combined loss gradient would reduce peak excess if waveform samples could be adjusted independently. The same result holds after propagating that gradient through the decoder's shared parameters for each individual example.

Some mel, feature-matching and adversarial components individually oppose peak correction. In the event-centered waveform probes, their total adverse contribution is at most 0.834% of the weighted waveform correction. This does not support the claim that adversarial loss overwhelms waveform supervision at these samples.

The 190 ms discriminator view can miss events, but placing it over an event is not an automatic remedy. The direction can change with surrounding context. These view probes hold the discriminator fixed and use a hypothetical singleton balancer observation; they are not a replay of the real training batch.

For quiet audio, all tested combined waveform directions point toward the teacher. One Odia case changes direction after passing through shared weights: contributions from its quiet interval are correcting, but contributions from the rest of the same recording outweigh them. That establishes local shared-parameter interference in this example, not an incorrect target or a universal quiet-loss defect.

## The native optimizer changes the peak direction

The native trace uses the actual updated discriminator, 32 training examples, preserved optimizer history, gradient clipping and loss balancing.

| Metric on the ten selected probes | Raw negative parameter-gradient direction | Native displacement, first-order direction | Measured change after the disposable update |
|---|---|---|---:|
| Full-scale peak-excess MSE | Improves | Worsens | **+5.93%** |
| Natural quiet residual MSE | Improves | Improves locally | **+3.09%** |
| Encoded-zero residual MSE, full six seconds | Improves | Improves locally | **+35.37%** |

For peaks, first-order analysis predicts a 5.77% increase, close to the measured 5.93%. This is mainly a direction problem for that update. Shrinking this fixed direction does not make it point downhill.

For quiet audio, the actual finite step exceeds the useful range of the local linear prediction. Squared-error curvature can produce this behavior even in a linear model; it is not evidence of a specific nonlinear layer failure. The six-second encoded-zero result includes startup and must not be conflated with the separate steady 2–6 second metric.

The raw gradient and native displacement have different norms, so their magnitudes are not directly comparable. The directional signs are the relevant evidence. No SGD step was executed, and these results do not establish that switching to SGD would improve quality or convergence.

### Which optimizer parameters contribute?

Additive first-order attribution assigns **78.42% of this predicted peak increase to AdamW-managed parameters and 21.58% to Muon-managed parameters**. Both routes' raw gradients initially favor reducing peaks. Both native route displacements favor increasing them.

The largest individual family contribution is the stem. Earlier blocks also contribute positively. Blocks 8 and 9 collectively offset some peak growth and provide most of the local quiet correction. This does not support freezing the late blocks based on older diagnostic results.

These percentages concern this update and these metrics. They are not general optimizer responsibility scores. Because both route contributions are positive, merely changing their two positive scalar learning rates cannot reverse this same first-order peak direction while holding all other update ingredients fixed.

### Momentum versus adaptive scaling

The AdamW formula was reconstructed from the actual post-update first and second moments, step counters, rates, epsilon and decay. This required no alternative optimizer update.

| Analytical direction on the same AdamW parameters | Peak direction |
|---|---|
| Current raw gradient | Improves |
| Accumulated first moment, without adaptive divisor | Worsens |
| Current raw gradient with the actual adaptive divisor | Worsens |
| Actual first moment and adaptive divisor | Worsens |

Both history and coordinate scaling can independently reverse the direction in this test. Resetting only first-moment history is therefore not supported as a sufficient remedy. The retained divisor still turns the current gradient uphill for peaks.

Decoupled weight decay slightly opposes peak growth. Floating-point reconstruction residual contributes only about 0.0042% of the AdamW route's predicted peak increase. The reconstructed update has 0.030% relative L2 error against the observed update, within the explicit FP32 rounding envelope. Rounding does not explain the direction reversal.

This isolates mechanisms, not a software defect. It does not prove the moments are corrupted or obsolete. Adaptive optimizers do not guarantee improvement of every held-out metric on every batch.

## Silence has two distinct residuals

The quarter-rate continuation remains the best tested 400-update continuation for overall reconstruction and silence. Natural quiet residual RMS falls from 0.000306715 to 0.000258495, and steady encoded-zero residual RMS falls from 0.000137479 to 0.000043793.

On digital silence, the remaining residual repeats exactly across 100 complete 40 ms cycles. Its power is 93.19% 480-periodic, 5.48% DC and 1.34% additional 1,920-periodic. The encoder's silence latent is nonzero; assuming that zero latent values represent encoded silence would use the wrong contract.

Natural quiet audio behaves differently. On 50 sources with enough complete quiet cycles:

| Residual component | Baseline share | Quarter-rate share | Absolute power change |
|---|---:|---:|---:|
| DC | 14.89% | 0.99% | −95.56% |
| 480-periodic | 18.06% | 8.87% | −67.27% |
| Additional 1,920-periodic | 6.41% | 8.21% | −14.58% |
| Varying with the input | 60.64% | **81.93%** | −9.93% |

These are energy components, not causal responsibilities. Both checkpoints use the same 2,563,200 unique samples for this decomposition. The varying component improves in all 50 sources, so there is no demonstrated plateau.

Subtracting the encoded-zero residual template from natural audio would improve quiet MSE by only 1.30%, equivalent to 0.65% RMS. It is not a useful general repair. Natural quiet student RMS is already close to the teacher: 0.000502568 versus 0.000505971. Quiet cosine improves from 0.8131 to 0.8686. The remaining issue is largely waveform reconstruction, not simply excessive volume.

Quiet pass/fail thresholds are strict engineering checks. Failing one does not itself establish an audible defect.

## Training data does contain digital silence

The actual 8.978 scored training hours contain approximately:

| Input level in 20 ms windows | Duration |
|---|---:|
| Exactly zero | 229.09 seconds |
| Nonzero RMS ≤1e−5 | 123.77 seconds |
| RMS 1e−5 to 1e−4 | 820.58 seconds |
| RMS 1e−4 to 1e−3 | 3,507.41 seconds |

There are 32 within-crop exact-zero runs of at least one second, across 26 sources; the longest is 2.56 seconds. Context and right-padding are excluded. Thus complete absence of silence is ruled out. This inventory does not prove the mixture is optimal, and source-input levels differ from teacher-output levels.

## Recommendation

Keep the step 8,890 checkpoint as the control and keep the encoder, teacher, decoder architecture and losses fixed. Do not interpret this audit as approval to restart long training with a new recipe.

The next comparison should isolate **optimizer state and update scaling** on several source-disjoint training batches, rather than adding a peak loss:

1. Compare the retained update with a freshly initialized optimizer-state reference and a small unpreconditioned gradient reference on disposable copies. Calibrate safe effective step sizes using training data; identical nominal learning rates are not equivalent steps. This distinguishes history effects from the update rule's coordinate scaling. Neither alternative has yet been tested as a training remedy.
2. Require favorable actual peak, natural-quiet and encoded-silence movement before choosing one optimizer-only short continuation. Preserve the current model and all existing reconstruction/expressive checks. A favorable single batch is not sufficient.
3. For silence, use measured effective displacement and a decreasing refinement schedule if supported by that comparison. The current constant quarter rate helped substantially, but the native trace shows that it can still take too large a step near a small residual.

This is a bounded test of the mechanism found here. Resetting one buffer, switching Muon off, freezing late blocks, adding a filter or increasing a loss weight would skip the causal evidence we now have.

A hard bound on every unseen output is a separate architectural requirement. A properly adapted bounded head can enforce it, but bounded output alone does not establish teacher fidelity. The earlier tanh experiment had quality tradeoffs, so it is not the first corrective change recommended by this audit.

## Evidence

- [Frozen waveform events and silence decomposition](waveform-diagnosis.json)
- [Existing loss-gradient directions](gradient-diagnosis.json)
- [Shared-parameter gradient geometry](parameter-geometry.json)
- [Native update with optimizer-state decomposition](native-chain-adamw.json)
- [Exact replay verification](native-replay-verification.json)
- [Actual training input inventory](input-inventory.json)
- [Frozen model preservation](complete.json)

No model weights, teacher targets or retained optimizer states were changed. The native probes used the H100 only for training diagnostics; these results are not CPU RTF measurements.
