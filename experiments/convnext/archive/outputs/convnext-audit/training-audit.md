# Convnext decoder training audit

9 September 2026. Completed checkpoint: step 10,000 on branch `Convnext`.

## Conclusion

There are substantive problems in the training objective and acceptance process. The run learned spectral structure, but the inspected outputs did not learn accurate teacher waveform reconstruction. Changing the learning rate alone does not address the measured loss imbalance.

The number near 22 was also misleading. Under the implemented objective, an exact copy of the teacher scores **21.6040** on the same 2,839 development clips. The final student scores **25.1725**. We should have measured that baseline before using total loss to judge convergence.

Earlier checks established numerical health, correct model updates and data accounting. I gave those checks too much weight when discussing quality progress. A successful real-audio waveform overfit test and loss-gradient calibration were missing before the extended run.

This audit used saved checkpoints, event logs, manifests and existing teacher targets. All new forward and gradient diagnostics ran on one CPU thread, with no optimizer steps. Training code, model weights and the production codec were not changed.

## What happened to the loss

The actual objective was:

```text
total = 15 × teacher log-STFT magnitude error
      +  1 × teacher waveform L1 error
      + 45 × original-recording log-STFT magnitude error below 8 kHz
```

Both spectral terms use uniformly averaged linear-frequency bins. Neither uses a mel filterbank. All quantities have their own scales, so the coefficients alone do not describe their influence on updates.

| Same 2,839 development clips | Exact teacher as prediction | Final student |
|---|---:|---:|
| Weighted teacher spectral term | 0.0000 | 3.7318 |
| Teacher waveform term | 0.0000 | 0.0299 |
| Weighted original-reference term | 21.6040 | 21.4107 |
| **Total** | **21.6040** | **25.1725** |

The encoder is lossy and the original decoder does not reconstruct the original recording exactly. Consequently, matching the teacher perfectly does not make the original-reference term zero. The teacher score is a useful baseline, **not a mathematical lower bound**: a student could trade some teacher error for a better match to the original recording.

The teacher baseline was recomputed on CPU from all existing dev targets and the exact leading-crop policy, with zero missing files. The student column is the completed run's final validation result using the same clips and objective. See [full baseline calibration](teacher-baseline-calibration.md) and [per-clip evidence](full-dev-teacher-baseline.json).

The training curve was not completely flat or always above 22:

| Training steps | Mean total loss |
|---|---:|
| 1,001–2,000 | 25.217 |
| 3,001–4,000 | 23.470 |
| 5,001–6,000 | 22.540 |
| 7,001–8,000 | 21.339 |
| 8,001–9,000 | 21.249 |
| 9,001–10,000 | 22.607 |

These are changing training batches, not repeated measurements on a fixed evaluation set. Many language queues emptied late in the run, so the final increase cannot be interpreted as model regression from this curve alone. All 9,000 new updates were logged without missing steps. [Curve evidence](final-curve.json)

## Confirmed findings

### 1. Waveform supervision was far too weak relative to spectral supervision

The audit measured actual weighted gradients, not just the sizes of loss values. On four held-out examples, the teacher spectral gradient at the final output-projection weights was approximately **15,700–22,700 times larger** than the waveform gradient. The original-reference spectral gradient was approximately **6,200–38,700 times larger**. These are diagnostic examples, not population estimates or final optimizer update ratios.

The independent output-waveform gradient check on eight examples showed the same imbalance. This means that a nonzero waveform term existed in code but contributed very little to the combined training signal. The planned gradient-contribution calibration was not performed before the long run.

On a deterministic diagnostic panel of 29 held-out utterances, evaluated at both the beginning and an interior position:

| Metric, 58 matched crops | Step 1,000 | Step 10,000 |
|---|---:|---:|
| Total loss | 32.688 | 24.847 |
| Teacher spectral error | 0.3587 | 0.2458 |
| Teacher waveform L1 | 0.02672 | 0.02734 |
| Mean student/teacher RMS ratio | 0.236 | 0.342 |
| Mean zero-lag waveform cosine | 0.0043 | 0.0113 |

Silence has waveform L1 **0.02594** on those same crops. The student's waveform error was slightly worse, while its spectral error improved. This is evidence of failed sample-aligned teacher imitation, not a claim that listeners would prefer silence. Low waveform correlation alone is not a perceptual quality metric.

A bounded ±40 ms lag search on the four gradient examples found maximum absolute correlation only 0.062–0.107. A small fixed timing shift does not rescue those examples. The inspected model produces substantially less energy than the teacher and does not reproduce its waveform accurately.

The evidence supports a phase/amplitude-supervision failure. It does not prove that the architecture is incapable of learning the mapping. [Matched diagnostics](completed-loss-diagnostics.json), [parameter gradients](parameter-gradients.json)

### 2. The implemented objective differs from the planned vocoder recipe

The plan specified an original-reference mel loss and subsequent adversarial and discriminator feature-matching training. The implementation used linear-frequency log-magnitude losses and retained reconstruction-only training through step 10,000. It has no adversarial, discriminator feature-matching or teacher-feature loss.

The published Supertonic reference uses mel reconstruction, multi-period and multi-resolution discriminators, and feature matching. Its coefficient 45 belongs to that specific loss definition. Moving the number to a different spectral reduction does not preserve its scale or behavior. This paper is architectural/training background, not a complete Supertonic 3 training release. [Supertonic paper, sections 3.1.2 and 4.2](https://arxiv.org/html/2503.23108v3#S3.SS1.SSS2)

Only the planned reconstruction stage was completed. That is insufficient evidence for the final perceptual-quality goal. Adding a GAN is a candidate continuation of the plan, not a proven cure for the current imbalance. The waveform-learning gate should be repaired first.

### 3. The preflight did not prove the required learning behavior

The small unit-test overfit check used a tiny synthetic model. The full decoder's short synthetic preflight reduced hybrid total loss from 63.63 to 11.62, but waveform L1 only moved from 0.10044 to 0.09727. Its original-reference branch was absent.

Those tests demonstrated wiring, gradient flow and spectral fitting. They did not demonstrate that this full decoder could accurately fit real AudioVAE2 teacher waveforms. The old 1,000-step speech run already showed a largely flat waveform term. That needed investigation before extending the same objective.

### 4. Acquisition diversity did not translate into the intended training mixture

The expanded phase consumed **371.70 unique scored hours** from **510.86 usable hours**, using exactly 576,000 nonoverlapping windows. Independent sampler replay reproduced the checkpoint state exactly.

However, equal language turns heavily favored FLEURS:

| Source | Available hours | Scored hours consumed |
|---|---:|---:|
| FLEURS | 340.20 | 322.35 |
| IndicVoices | 44.04 | 40.61 |
| LibriSpeech | 100.00 | 1.99 |
| All expressive sources | 26.69 | 6.75 |

All English sources together received only about 4.01 hours, or 1.08% of exposure. FLEURS contributed 86.72%. Thus the downloaded source quotas were not the achieved training weights. This is a mixture-design problem, although it does not by itself explain the earlier loss plateau. [Data and sampler audit](data-sampling-validation-audit.md)

### 5. Validation did not cover the target deployment population

The final dev set has 763 English-labeled clips, 531 Japanese-labeled clips and 1,545 with unspecified language. It contains **zero FLEURS and zero IndicVoices dev clips**. EmoGator accounts for 53.89% of the equally weighted utterance mean; only 25 clips are LibriSpeech.

Evaluation uses each clip's first up-to-2.56 seconds. It does not establish quality across the requested languages or interior streaming behavior. There was no teacher baseline or step-1,000 comparison on this same full panel before the audit. The earlier 25-clip English validation and the expanded final validation are not directly comparable.

## What the audit ruled out

- The teacher is the pinned original AudioVAE2, with frozen parameters, raw 64-channel posterior means at 25 Hz and fixed 48 kHz decoding. There is no missing stochastic noise condition in this teacher configuration.
- No identified 16/48 kHz conversion, latent scaling, crop-offset or phase-shuffle indexing bug.
- The teacher decoder needs at most 20 previous latent frames; the student has 29. Independent CPU context checks matched cropped and whole decoding exactly.
- Losses exclude padding and context-only samples; the new batch loss is an example mean rather than an accidental sum.
- Muon and AdamW states were carried over correctly. Both cover the intended disjoint student parameters. All blocks changed, and trained LayerScale values grew well above initialization.
- No observed missing updates, nonfinite optimizer state, frozen model, corrupted cache identity or repeated new-phase scored windows. These checks do not rule out every possible learning problem.

The constant learning rate, lack of scheduling and eightfold batch-size change remain optimization variables. They are secondary hypotheses; no evidence identifies them as the principal explanation for the measured waveform-gradient imbalance. [Optimizer/objective audit](objective-and-optimizer-audit.md), [alignment audit](latent-target-alignment-audit.md)

## Recommended next experiment

1. Preserve this checkpoint and its evidence. Do not extend the same recipe just to force the total below an arbitrary number.
2. Define success on identical teacher/student clips: separate teacher distance, original-reference distance, amplitude, phase-sensitive reconstruction and perceptual quality. Retain a teacher baseline and simple controls.
3. Correct the loss design and calibrate actual gradient contributions. Make fullband teacher imitation the explicit primary task. Treat the original 16 kHz reference as a separately justified auxiliary objective. Evaluate suitable energy-normalized waveform or complex-spectral supervision; do not blindly multiply the waveform weight by the measured gradient ratio.
4. Demonstrate accurate fitting on a tiny fixed set of real teacher examples with the full decoder, including voiced speech, fricatives and expressive sounds. This diagnostic requires repeated examples and should remain separate from the fresh-data main run. It must be explicitly agreed as an exception to the main run's no-repeat policy before execution.
5. Once that gate passes, add the planned perceptual training components in a controlled comparison. A frozen teacher provides targets; it does not transfer its learned decoder weights or perceptual prior automatically.
6. Use declared source/language/condition exposure weights and a representative, speaker-separated held-out panel with interior crops. Reuse the remaining fresh windows where appropriate. Only then choose learning-rate scheduling and a longer training budget.

No replacement decoder or optimizer switch is justified by this audit alone. The first priority is to show that the existing architecture can learn the intended waveform under an appropriately balanced objective. This report recommends changes; it does not implement or start them.
