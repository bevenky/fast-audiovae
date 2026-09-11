# Completed diagnostic pilot and next steps

Reviewed 9 September 2026. The 1,009-update run is a diagnostic pilot while the recipe is being developed. The final model training run has not started. No new training was launched during this review.

## What the completed pilot establishes

The run stopped normally after 32,275 unique scored windows, totaling 20.007 hours. It used a fresh student, the frozen original AudioVAE2 encoder and decoder teacher, raw 64-channel latents, and Muon plus AdamW. Saved source identities match the checkpoint. The teacher is unchanged, the hidden layers are updating, and there is no evidence of a plateau at the budget boundary.

| Development measure | Update 200 | Update 800 | Update 1,009 |
|---|---:|---:|---:|
| Mean nonquiet waveform cosine | 0.073 | 0.378 | 0.525 |
| Normalized waveform error, lower is better | 1.300 | 0.577 | 0.490 |
| Teacher mel error, lower is better | 3.343 | 1.714 | 1.649 |
| Mean quiet residual RMS | 0.020280 | 0.001615 | 0.001179 |

The fixed panel has 84 recordings and 167 crops. Two globally quiet crops are excluded from cosine. No remaining crop reaches 0.99, and all 1,540 quiet 20 ms windows fail their strict residual criterion. These are reconstruction diagnostics, not percentages of perceptual quality. The quiet limits are provisional engineering tolerances, not calibrated listening thresholds.

## Findings that should drive the next experiment

1. **Active audio is attenuated.** Median output is 9.84 dB below the teacher; 164 of 165 nonquiet crops are more than 1 dB too quiet. Pauses simultaneously contain too much energy. Raising global output gain would worsen the pauses.
2. **The effective loss balance differs from its description.** The intended final mel contribution is 25%, but its measured mean pre-sum gradient share is 43.42% over updates 751–1,009, reaching 71.46% on some batches. The 0.999-decay gradient history lags current gradients. This is the behavior of the formula, not a numerical failure, but it does not enforce the intended cap. It has not yet been proven to cause the attenuation.
3. **The quality-training stage is unused.** MPD/MRD adversarial and feature-matching losses are implemented but have never updated this model. The current entry gate inherits final reconstruction acceptance and can prevent that stage from being tested at all.
4. **Data exclusions were unnecessarily global.** Audio seen by a discarded older model was excluded from the new model too. This removed useful languages and explicit vocal events. The user's clarified policy restores that training pool across independent diagnostics.
5. **Expressive reconstruction is much worse than speech.** Mean cosine is 0.626 for speech, 0.365 for laughter, 0.0057 for screaming and 0.0354 for whistling. These groups need separate results. Generic emotion labels are not substitutes for confirmed crying or giggling.
6. **Quiet output has a repeating component.** A bounded 12-crop waveform diagnostic found a shared 480-sample template in several quiet clips. In three examples with enough quiet windows, it explains about 29–52% of student AC power, versus about 0.6–4.3% for the teacher. The period matches the direct waveform head's 10 ms frames. This is evidence consistent with a projection artifact, not yet proof that one particular layer causes it. The final projection has no bias; constant intermediate features could still project into such a pattern. It is much larger than a simple DC offset. A bounded timing-shift search does not rescue screaming or whistling, and ordinary speech has mostly zero or two-sample best shifts.

Quiet training data is present: approximately 38.7 minutes of scored teacher output below -80 dBFS and 114,642 quiet transitions. More silence alone is not an established fix. Beginning and interior crop statistics are similar, so saved aggregates do not point to a failure confined to crop starts. The new waveform diagnostic also finds strongly attenuated speech bands and poor higher-frequency correlation; this is broader than output gain alone.

## Data policy for the next diagnostics

- Reuse downloaded training audio across independent diagnostic experiments, including material used by discarded older models.
- Within a run and its exact continuation, never repeat a scored audio interval. In a fork of this checkpoint, inherit this student's consumed intervals. A/B arms may each consume the same comparison sequence once.
- Keep development and sealed test data excluded by source, hash, parent recording and available speaker/session identity. Never turn the fixed development panel into training data.
- Overlapping causal context is required for correct decoding; it is not additional scored training exposure. Repeated evaluation is also not a training repeat.
- Preserve old manifests and ledgers as historical evidence. Create a new versioned comparison manifest rather than erasing previous exposure.

The downloaded inventory contains all 22 scheduled Indic languages and the requested Chinese, Arabic, Spanish, Portuguese and French material. It also contains explicitly labeled laughter, crying, giggling, screaming, shouting and human whistling. The live metadata audit finds **481.688 source hours** available for this student's continuation after conservatively excluding its whole pilot recordings and held-out identities. Exact scored-window capacity remains to be allocated. See [the data audit](data-next-phase.md). The earlier roughly 110-hour remaining-capacity restriction was a consequence of the superseded global exclusion policy.

A preparation defect found in this review has been fixed: the small expressive supplement used pretty-printed concatenated JSON objects under `.jsonl` names. A new `versions/v3-jsonl` now passes the standard manifest loader and validation, preserving all 26 training and 8 development records, their fields and order. All 68 original/prepared audio hashes were rechecked unchanged, and the archived version remains intact. [Repair evidence](expressive-manifest-repair.json). This does not change the active development panel or training plan.

## Proposed execution order

### 1. Preserve the baseline and use the completed output diagnostic

The 12-crop diagnostic completed without changing model or teacher weights. It compared unshifted amplitude, frequency-band energy, bounded diagnostic lag and quiet residual structure. Original held-out cache records had been evicted, so only selected whole-utterance targets were regenerated with the pinned FP32 teacher. Byte differences from the original batched targets are explicitly recorded; this diagnostic does not replace the saved acceptance evaluation. No fitted shift, gain or subtraction is applied to the reported acceptance results. Checkpoint SHA256: `955d12a84245765331b8697747e1565a3a90f3f2e1f6d8819e1c4f62d99ac104`.

### 2. One matched loss-balance comparison

Fork the completed checkpoint into two diagnostic arms with identical optimizer state, normalization, learning rate, data order and teacher targets:

- **Control:** the current EMA-based balancing formula.
- **Candidate:** an explicitly measured current-batch cap of 25% on the mel gradient contribution. Redistribute contribution to waveform anchoring while preserving the total pre-sum norm budget where both gradients are nonzero. Log zero-gradient and coefficient-bound exceptions; do not claim a cap when it cannot be satisfied.

Use 500 updates per arm, evaluating at the start, 250 and 500. At the previous mixture's measured consumption this is approximately 9.9 scored hours per arm; the exact duration depends on the new windows and is not known until the manifest is built. Both arms may share those audio hours because they are separate experiments. Do not extrapolate GPU utilization or full job duration from training-loop time alone.

Target roughly 30% English, 30% Indic across all 22 languages, 35% other languages and 5% explicit vocal events by scored duration, subject to verified source capacity. Include Chinese, Arabic, Latin American Spanish, Brazilian Portuguese, French and Japanese. Cap vocal-event quotas at available unique audio rather than replaying scarce examples. Quiet intervals and natural transitions should remain in the speech and expressive categories.

Do not simultaneously change the architecture, optimizer, normalization freeze or quiet-loss coefficient. The previous 10% quiet-loss candidate failed its declared comparison and remains rejected.

Provisional promotion rule, to be frozen with the experiment manifest: at least 10% lower normalized waveform error than control at update 500, output-level error moving toward the teacher, and no more than 5% regression in mel error or mean/p95 quiet residual. Inspect every event group and zero/clipping counts; aggregate improvement must not conceal collapse of a class. The update-250 comparison should support the same direction. If the result is mixed, keep the control rather than presenting the candidate as a fix.

If the repeating quiet component persists, test a targeted, training-only penalty on the residual's mean 480-sample pattern over sufficiently long, teacher-identified quiet regions. Compare the student pattern to the teacher pattern, preserving real room tone and breath. First measure its gradient contribution and run one matched control comparison; do not add it simultaneously with the balancing change. Do not subtract a fitted template at inference, mute pauses or remove intermediate affine terms without demonstrating causality and quality preservation. This is a new candidate informed by the diagnostic, not the previously rejected general 10% quiet loss.

### 3. Complete the perceptual recipe as a separate bounded trial

Separate **readiness to test perceptual training** from **final acceptance of the trained decoder**. Readiness should require correct alignment and masks, frozen teacher identities, finite gradients and outputs, consistent streaming state, and a stable reconstruction baseline. Record the checkpoint, panel and readiness evidence. Keep the user's 0.99 reconstruction target and local quiet tests visible as final goals; do not quietly lower them.

Once the reconstruction comparison and output diagnostic support proceeding, fork the selected checkpoint into reconstruction-only control versus a gradual MPD/MRD adversarial plus feature-matching trial. Keep the waveform anchor. Use an initial budget of 1,000 updates per arm with a safety review at 250; define rollback limits before launch. Discriminators are newly trained against original-teacher waveforms, not pretrained AudioVAE2 discriminator features. Specify their optimizer configuration explicitly before the first update.

This is supported by the [Supertonic paper's autoencoder training objective](https://arxiv.org/html/2503.23108v3#S3.SS1.SSS2), which combines mel reconstruction, adversarial and feature-matching losses. Its reported 1.5 million updates provide context that 1,009 updates is early, not a required budget for this distillation or an exact Supertonic 3 training recipe. These losses and discriminators add training work without adding decoder inference layers.

### 4. Settle the recipe before final training

Choose from matched evidence, then freeze the data split, loss rules, normalization schedule and optimizer settings for final training. Preserve per-language and per-event reporting and add the missing held-out language/event groups as a separately versioned panel. Keep the current panel unchanged for comparison. Run listening and the agreed speech-quality measures when reconstruction is useful enough to interpret them; speech MOS predictors alone cannot qualify whistles or crying. Qualify CPU streaming RTF on Apple, Intel and AMD only on a quality-qualified candidate.

## Supporting evidence

- [Saved metric audit](saved-metrics-audit.md)
- [Recipe and checkpoint audit](recipe-audit.md)
- [Complete scalar summaries](completed-metrics-summary.json)
- [Per-crop scalar evidence](completed-crop-metrics.jsonl)
- [Bounded waveform diagnostic](waveform-audit.json)
- [Waveform interpretation and provenance limits](waveform-findings.md)

No architecture change, new inference layer, code commit or release is claimed by this plan.
