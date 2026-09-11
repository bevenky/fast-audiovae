# AudioVAE2 student: recipe v2 audit

**Recommendation: continue this run. The audit found important quality and coverage gaps, but no new structural training failure that warrants restarting from scratch.**

The preserved step 1,700 checkpoint is learning substantially better than at step 1,000. All four previously identified training defects have been addressed in the active recipe. The remaining shortcomings need explicit acceptance checks and broader validation, rather than another speculative learning-rate or architecture change.

This audit inspected source, saved training metrics, the immutable sample plan, calibration records and the actual trained checkpoint. It evaluated the fixed development panel on CPU and checked trained-model streaming parity. It performed no optimizer updates, RTF benchmark, source change or training restart. Audio and checkpoints remain on Runpod. The step 1,700 checkpoint was preserved there before evaluation.

The final live status read showed update 2,157, exactly 1,657 discriminator updates, and the training process still active. The instantaneous GPU sample was 100% utilization with 77,109 MiB of 95,830 MiB used. Free disk space was 6.36 GiB. These are status observations, not a throughput benchmark or evidence of quality beyond the evaluated step 1,700 checkpoint.

## What the metrics show

The same 167 source/start pairs and 16,465,920 scored samples are used below. Cosine is waveform similarity to the frozen teacher, not a perceptual quality score such as PESQ or UTMOS.

| Fixed development measure | Step 500 | Step 1,000 | Step 1,700 |
|---|---:|---:|---:|
| Speech waveform cosine, higher is better | 0.268 | 0.683 | **0.824** |
| All active audio cosine, higher is better | 0.221 | 0.569 | **0.701** |
| Normalized waveform error, lower is better | 0.930 | 0.487 | **0.338** |
| Teacher mel error, lower is better | 2.953 | 1.997 | **1.527** |
| Mean quiet residual RMS, lower is better | 0.010597 | 0.002653 | **0.000539** |
| Median repeating residual at the 480-sample output period | 0.009937 | 0.001396 | **0.000268** |
| Samples exceeding full scale | 0 | 0 | **75** |

From 1,000 to 1,700, waveform error falls 30.7%, mel error 23.6%, and quiet residual 79.7%. Waveform error and cosine improve on 166 of 167 crops; mel improves on 165. This is broad improvement, not a few favorable examples moving the average.

Step 1,700 has consumed approximately 35.00 unique scored hours. The previous r7 control consumed approximately 34.95 hours. At this similar exposure, the new model has 4.1% lower waveform error and higher cosine, but its mel error is still 6.7% worse and quiet residual 42.7% worse. This is a mixed quality result, not a universal win. Different initialization, recipes and data order prevent attributing that comparison to one fix.

The new evaluation used CPU; earlier saved evaluations used CUDA. Panel identities and scored lengths match. The maximum difference in reported teacher RMS is 2.98e-8, far smaller than the observed changes. This audit did not establish a new CPU-versus-GPU numerical equivalence result.

## The four prior defects are addressed

| Previous issue | What is verified in this run |
|---|---|
| GAN and feature matching waited for near-perfect reconstruction | Both start at update 501, ramp over 500 updates and reach the declared full mix at 1,000. The saved step 1,700 checkpoint contains exactly 1,200 discriminator updates. The 0.99 goal is not an entry requirement. |
| Planned fixed-weight normalization calibration was missing | Calibration ran after update 500 on 512 separate training windows, with two sequential passes and zero optimizer updates. Both passes used identical inputs. The actual checkpoint retains the calibrated frozen buffers. |
| Quiet clips and short tails received disproportionate waveform weight | Waveform MAE pools valid samples; mel pools valid time-frequency elements; discriminator and feature objectives use valid-duration weights. The inverse-RMS values are detached diagnostics only. Partial tails remain valid data. |
| Each phase of the learned latent adapter was poorly conditioned | The current adapter repeats the original 64-dimensional latent and adds a learned phase bias. Each phase preserves an identity Jacobian with respect to the raw latent; it no longer relies on a learned matrix to retain all dimensions. |

The original encoder and decoder teacher remain frozen. The student consumes the same detached 64-channel encoder latents and learns against the teacher's decoded audio. Teacher inference is batched during target preparation. The student does not need to learn an alternative latent space.

All inspected numeric training fields and checkpoint parameters are finite. Muon owns the 20 hidden matrix weights; AdamW owns the remaining student parameters. Discriminator parameters have their separate optimizer. Discriminator gradients are disabled for its parameters during the generator update while the gradient through its input audio remains connected. No loss branch has a zero output gradient or hits the balancer's scale cap.

The checkpoint loads through the strict production recipe loader. Normalization variances and effective affine scales are well behaved, with no collapsed channel found. Five real-condition tests covering speech, laughter, screaming, whistling and quiet audio pass one-latent streaming versus batch and normalization-fold parity. The largest streaming difference is 8.35e-7; folding differs by at most 3.06e-7. These are correctness checks, not speed or all-condition quality claims.

## Confirmed quality gaps

### Expressive sounds remain much weaker than speech

At step 1,700, mean cosine is 0.589 for laughter, 0.066 for screaming and 0.068 for whistling. The whistle group is only two recordings/four crops, so this is a clear failure on those examples, not a population-wide estimate.

Low output level is not a simple global gain bug. Using the saved statistics, the best possible per-crop scalar gain would reduce waveform MSE by only 0.291% at the median. Boosting outputs to match teacher RMS would amplify unmatched content and would not improve cosine. Missing content, timing or phase needs to be learned; a volume patch would not solve it.

### Quiet audio improves but still fails the current acceptance checks

All 1,540 quiet windows still fail the provisional strict checks, despite the large drop in residual noise. A repeating component at the 480-sample output period also remains. Earlier analysis found it in both beginning and interior crops, so it is not confined to stream startup. The audit does not establish which layer causes it.

The quiet thresholds are engineering checks, not calibrated audibility judgments. Reporting only a zero pass rate hides genuine progress. Actual residual RMS, repeating residual, output level and their worst cases should accompany that count. No active crop yet reaches the final 0.99 cosine target.

### Two laughter crops overshoot full scale

| Crop | Student peak | Teacher peak | Student samples above full scale |
|---|---:|---:|---:|
| freesound:240901, start frame 0 | 1.3980 | 0.9910 | 74 / 30,720 |
| freesound:25794, start frame 45 | 1.0319 | 0.9948 | 1 / 122,880 |

Teacher samples never exceed full scale in either crop. The overshoots occur inside the scored crops, outside their first and last 10 ms. This is real transient overshoot, not merely padding at a crop boundary. The direct waveform head is unbounded. The count measures excursions in raw output, not samples already clipped by an audio-file writer.

The existing final reconstruction gate does not explicitly reject these excursions. Add peak behavior to final acceptance reporting. Do not hide the issue with a playback clamp or add a new output activation without evaluating its effects. The current evidence does not show that this early localized overshoot requires a fresh model.

## Data and validation gaps

All 320,512 optimization-plus-calibration intervals passed the combined identity and scored-overlap checks. The inspected 1,497 journal records exactly match their planned batches, cursor and hash chain. Different scored crops from the same file are allowed; scored intervals do not repeat. Shared causal context is intentional. Original hash and decoded-length checks occur when targets are prepared; this audit did not rehash every audio file independently.

The first 1,000 updates contain 20.58 unique scored hours. The full 10,000-update plan contains 206.54 hours, not 500 hours of scored optimization. Downloaded corpus size and the audio actually consumed by this schedule are different quantities.

| Duration share | Full run plan |
|---|---:|
| English | 31.29% |
| Indic | 31.06% |
| Other identified languages | 37.09% |
| Dedicated explicit events | **0.56%** |

Training covers 110 languages, including all 22 scheduled Indian languages. However, the dedicated event share is far below the intended 5%. Across the entire run, labeled crying contributes only 141 seconds, giggling 122 seconds, human whistling 112 seconds, shouting 41 seconds and yelling 4.8 seconds. These are scored windows from labeled recordings, not densely annotated event durations. Rare-event scarcity is a plausible contributor to the weak results, not a proven sole cause.

The current validation panel contains 84 recordings and 18 identified speech languages, with only 13 of 22 Indic languages. It lacks Bodo, Dogri, Konkani, Kashmiri, Maithili, Manipuri, Odia, Sanskrit and Santali, as well as Chinese, Arabic and German. It also lacks dedicated crying, giggling, shouting, whispering and breathing groups.

A read-only inventory of 5,631 existing declared dev/test candidates found seven usable additions: four whispering and three breathing recordings, totaling 50.59 seconds. They are distinct from the current panel and disjoint by known identity/hash/parent metadata from the entire optimization and calibration source set, including future training windows. Keep them as a separate appendix so the existing learning curve remains comparable. Their actual audible content and decoded lengths still need verification at evaluation time.

There are no existing declared reserved candidates for the other listed gaps. Filling them requires additional independently reserved data. Do not move future training rows into validation or repeat rare training intervals to manufacture a larger event share. Unknown speaker identities in parts of the corpus remain a split limitation.

## Training-policy and monitoring issues

1. **The `train/total` chart is incomplete.** It is raw waveform plus mel, excluding adversarial loss, feature matching and adaptive balancing. It should be labeled reconstruction loss; component losses and gradient shares need separate interpretation. A GAN's moving objective is not expected to approach a single teacher score.
2. **The adaptive balance is not currently realizing the nominal proportions.** In the inspected recent window, individual scaled output-gradient norm magnitudes are approximately 17% waveform, 50.5% mel, 20.8% feature matching and 11.6% adversarial, compared with nominal 30/40/20/10. The EMA decay of 0.999 lags changing norms. These are component norm fractions, not projections onto the combined gradient or parameter-update shares. This is worth tracking, but continued heldout improvement does not justify another arbitrary cap or weight change now.
3. **Gradient clipping is active on every inspected generator step.** The median norm before clipping fell from roughly 4,075 before calibration to 162 in the recent window; discriminator norms also clip. Muon and AdamW transform gradients, so clipping factors alone do not establish tiny parameter updates or optimizer failure. There is no numerical divergence or dead objective branch in the inspected records.
4. **Calibration is valid but not distribution-identical.** Small-group minimums give its 20.5 minutes proportionally more other-language and event data than training. This could affect normalization quality, but checkpoint statistics are healthy and there is no measured defect that warrants recalibrating the live model arbitrarily.
5. **Checkpoint and evaluation coverage should be stronger.** The ordinary schedule has a large gap between step 1,000 and 5,000. This audit fills it at 1,700. The normal latest checkpoint is overwritten; the audit preserved 1,700 explicitly. Retain a bounded set of milestone/best checkpoints for comparison and controlled continuation. Storage is limited, so retention must be bounded.

The objective-routing review is consistent with the training-only loss roles in the [Supertonic paper](https://arxiv.org/html/2503.23108v3#S3.SS1.SSS2), [Vocos training implementation](https://github.com/gemelo-ai/vocos/blob/main/vocos/experiment.py) and [EnCodec gradient balancer](https://github.com/facebookresearch/encodec/blob/main/encodec/balancer.py). Those references do not guarantee this student's convergence or justify importing their settings blindly.

## Next actions, in order

1. **Keep the current weights, optimizer state and recipe.** At step 1,000 the full perceptual ramp had only just completed. The additional full-objective updates are producing substantial gains. No from-scratch restart is supported by this audit.
2. **At the next review, use the same fixed panel and inspect the weak conditions and peak excursions separately.** A milestone around step 3,000 is more informative than repeated short tuning trials. The 0.99 target remains final acceptance, alongside quiet, level, peak and perceptual/listening checks.
3. **Expand validation without changing the historical aggregate.** Add the seven reserved whisper/breathing recordings as a separate panel, then obtain independently held-out material for the missing languages and conditions.
4. **Increase unique expressive coverage for a subsequent training stage.** The current 0.56% share cannot test the intended broad expressive capability. Extra examples must respect the within-run nonrepetition rule.
5. **Improve observability and bounded checkpoint retention at a safe instrumentation update.** Rename the incomplete total, surface quiet/periodic/peak metrics, and distinguish scheduled from realized gradient shares. These changes do not require new weights.
6. **Only make a controlled recipe change if fixed-panel progress stalls or a defect is demonstrated.** If the persistent balance discrepancy coincides with a waveform/quiet plateau, compare one justified balancing change from a saved checkpoint. Do not combine a learning-rate change, new activation, new loss and recalibration in another unexplained restart.

No claim of teacher-level quality, all-language generalization or improved CPU RTF is justified yet. The current evidence supports continued training with a more complete acceptance and coverage plan.

## Evidence

- [Saved-metrics review through step 1,522](metrics-review.md)
- [Step 1,700 fixed-panel and gain analysis](current-evaluation-review.md)
- [Checkpoint, calibration and streaming audit](checkpoint-audit.json)
- [Step 1,700 evaluation summary](current-evaluation-summary.json)
- [Laughter peak verification](peak-audit.json)
- [Data and calibration review](data-audit.md)
- [Reserved validation availability](reserved-panel-availability.md)

Preserved checkpoint SHA256: `e2d4770eac1ac2f42e1812e5094cec2587597a08eb20c5927078bf3fb3b7386f`.
