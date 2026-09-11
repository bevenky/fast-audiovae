# Completed pilot: independent training recipe audit

This audit read the live Runpod r5 checkpoint, saved evaluations, all 1,009 training metrics, and the deployed training source. No inference, training or weight changes were performed.

## Result

The run finished its declared budget normally: 1,009 updates, 32,275 scored windows and 20.00708 unique scored hours. Its accumulated training-loop time was 1,146.18 seconds, which excludes scheduled development evaluations and some setup/checkpoint work. It is stopped awaiting review.

| Held-out metric | Update 200 | Update 600 | Update 800 | Update 1,009 |
|---|---:|---:|---:|---:|
| Mean nonquiet cosine | 0.07288 | 0.27827 | 0.37790 | 0.52541 |
| Mean normalized waveform error | 1.30025 | 0.61036 | 0.57748 | 0.48996 |
| Mean teacher mel error | 3.34324 | 2.05144 | 1.71401 | 1.64926 |

No held-out nonquiet crop reaches 0.99. All 1,540 quiet windows fail the declared strict local criteria. The model remains unqualified, but its held-out correlation is still improving rather than plateauing at the budget boundary.

## Verified mechanics

- All 14 deployed implementation files match their checkpoint identity hashes.
- Frozen original AudioVAE2 source/checkpoint identities are retained. Teacher parameters are excluded from optimization and encode/decode run without gradients. Student inputs remain raw 64-channel posterior means, with 1,920 output samples per latent. There is no new latent normalization or latent-target substitution.
- The student was initialized fresh for this representative pilot, as declared. The earlier fitted-crop diagnostic weights were not used. This 1,009-update run therefore is not an additional 1,009 updates on top of the successful fitted-crop model.
- Twenty hidden matrices use native Muon; the other 82 parameter tensors use AdamW. All saved AdamW update counters equal 1,009. The learning rate warmed up for 50 steps and remained 0.0002 afterward.
- Both masked normalization sites have exactly 200 statistic updates, then persistent frozen statistics. Affine parameters remained trainable. This follows the declared recipe.
- No adversarial or feature-matching training occurred. `perceptual_start` is null and the discriminator optimizer has no state entries. Those losses are implemented but have not contributed to these weights.
- The extra quiet loss is disabled, consistent with rejection of its earlier 10% calibration setting. Quiet-window evaluation is enabled.

## Concrete objective accounting gap

The final curriculum is configured as 75% waveform and 25% mel, but the balancing coefficients use an exponential moving average with decay 0.999. This is a long history compared with a 1,009-update pilot. As the raw mel gradient changes, its actual scaled share differs substantially from its configured share.

The actual pre-sum mel share below is computed for each batch as `scaled_mel_norm / (scaled_mel_norm + scaled_waveform_norm)`, then summarized across batches. It is not a parameter-update share or an attribution of perceived audio quality.

| Updates | Configured final mel share | Actual mean | Actual median | Actual range |
|---|---:|---:|---:|---:|
| 501–750 | 25% | 42.05% | 41.89% | 16.56–68.05% |
| 751–1,009 | 25% | 43.42% | 43.19% | 19.21–71.46% |

No balancing coefficient hit its min/max bound. This is expected behavior of the current EMA formula, rather than a silent numerical failure, but describing this pilot as receiving only a 25% mel contribution would be inaccurate. The gap warrants a controlled comparison if the intended curriculum requires that bound.

## Clipping does not show that the hidden layers stopped learning

Every late update was globally clipped to norm 1.0. However, Muon and AdamW transform those gradients, so the clipping multiplier cannot be treated as a matching reduction in weight movement.

At update 1,000, measured relative parameter changes were 0.639% for the adapter, 0.126% for the middle hidden projection and 0.0831% for the output projection. The hidden blocks' learned layer-scale mean magnitudes reached 0.024–0.032, compared with their 0.000001 initialization. The decoder's hidden layers are active and updating. These observations do not justify disabling clipping or blaming an optimizer freeze.

## Gaps that remain hypotheses

- Freezing normalization at update 200 is an engineering choice, not a demonstrated optimum. There is no observed abrupt development failure after it, so changing normalization together with the objective would make the next result hard to interpret.
- Twenty hours used once provides only 1,009 weight updates. That is not evidence of convergence for a fresh waveform decoder. Late training waveform error averages about 0.511, while the fixed development panel is about 0.490; these are different mixtures, so the values are not a matched generalization estimate, but they do not show a large memorization gap.
- The present architecture is not proven inadequate by this short run. The fixed-crop experiment already demonstrated high reconstruction accuracy with this architecture; the current broader run is still learning.
- The 0.99 correlation and strict local quiet criteria establish an exact reconstruction goal, not a general perceptual-quality score. We should preserve them as visible acceptance evidence without claiming a specific MOS from them.

## Recommended next bounded comparison

1. Preserve the completed checkpoint, optimizer states and consumed-source ledger. Continue on unused balanced recordings rather than discarding the learned state and restarting.
2. Use the same fresh window sequence for a two-arm comparison: the current objective as control, and a variant with an explicit current-batch mel-share bound and measured contribution logging. Hold architecture, optimizer, learning rate and frozen normalization constant. A shorter EMA is another candidate, but changing both controls at once would weaken attribution.
3. Decide on the variant from fixed development waveform, mel, amplitude and quiet residual metrics, including nonverbal groups. A better waveform correlation accompanied by worse silence or spectral quality is not sufficient.
4. Define separate, recorded readiness criteria before entering adversarial/feature-matching training. Currently `enable_perceptual()` requires the exact model to pass the same gate containing 0.99 for every nonquiet evaluation crop and every local quiet-window check. That can indefinitely prevent the intended perceptual stage. Separate readiness from final acceptance rather than quietly lowering the final goal. This audit does not recommend automatically enabling the GAN on the current unqualified checkpoint.

Source evidence: deployed `audiovae_student/gradient_balancer.py` lines 18 and 105–145; `distillation_training.py` lines 276–290, 335–355 and 557–584; `teacher.py` lines 88–92 and 216–237; saved r5 `latest.pt`, `metrics.jsonl`, and `evaluation-step*.json` under `training-runs/representative-reconstruction-v1`.
