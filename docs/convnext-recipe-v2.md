# Corrected frozen-teacher decoder recipe

This is a fresh student on the `Convnext` branch. The previous checkpoints remain available for comparison. Changing the latent adapter and training objective is not an exact continuation of those models.

| Audit finding | Change |
|---|---|
| GAN and feature matching never trained | Waveform and mel start at update 1. After 500 reconstruction updates and fixed-weight calibration, discriminator updates start at update 501. Generator GAN and feature-matching contributions ramp over 500 updates. |
| Normalization frozen from an early moving average | Stop optimizer updates at the warmup boundary. Measure sample-weighted population moments on a separate, representative training-only calibration panel. Calibrate the stem first, freeze it, then calibrate the final normalization under that stem. Both sites remain fixed afterward; their affine parameters stay trainable. |
| Quiet clips and short tails had excessive relative weight | Waveform supervision uses total absolute error divided by the total valid sample count. Mel losses pool valid time-frequency elements separately at each resolution. There is no inverse per-clip RMS multiplier. Context and artificial padding are excluded. Fixed-length discriminator crops also receive valid-duration weights so short tails do not regain excess weight in GAN or feature matching. |
| Per-phase latent matrices poorly conditioned | Each 40 ms raw latent becomes four representations `z + phase_bias`. Each phase has an identity Jacobian with respect to all 64 original coordinates. Only the offsets are learned. The existing 100 Hz ConvNeXt body and direct waveform head remain. |

## Schedule

Plan one 10,000-update run on the existing H100, with batch size 32 and distinct scored audio intervals. Use the existing Muon plus AdamW student optimizer; this is not another optimizer comparison. Keep the original AudioVAE2 encoder and decoder frozen and use the same deterministic raw posterior means and continuous teacher waveform targets.

- Updates 1–500: waveform and multi-resolution mel reconstruction, with masked training normalization. Learning-rate warmup occupies the first 50 updates.
- Boundary after update 500: two calibration passes with weights fixed. The second pass uses the calibrated, frozen stem. No development audio is used for calibration.
- Updates 501–1,000: train all five period and three resolution discriminators, ramp GAN and feature matching into the generator objective.
- Updates 1,001–10,000: continue the complete objective. Save checkpoints and expose progress in TensorBoard. Do not stop merely because waveform correlation has not reached 0.99.

Generator output-gradient shares start at 50% waveform and 50% mel and transition to 30% waveform, 40% mel, 20% feature matching and 10% adversarial. These are declared engineering choices using the existing gradient balancer, not the published paper's scalar loss coefficients. A target share is not a guarantee of its instantaneous realized gradient share; realized norms and scaling limits remain logged. Discriminator AdamW uses learning rate 0.0002, betas (0.8, 0.9), and zero weight decay. The exact recipe is versioned in checkpoints.

## Data and evaluation

Use downloaded training audio on Runpod. Reuse across independent debugging runs is allowed; the new run must not repeat scored training intervals. Reserve separate calibration windows. Replaying those windows twice estimates statistics without optimizer updates. Preserve all existing validation and test reservations.

Allocate feasible language and expressive quotas before ordering the run, then spread each subgroup across it. This prevents scarce events from being consumed only at the beginning. Report actual hours and coverage rather than implying equal coverage of every language or expression.

Keep the original development panel unchanged for comparisons. Its 18 identified speech languages do not establish quality across every training language. Track historical normalized waveform loss and mel loss under separate diagnostic names, alongside correlation, level, clipping, quiet residuals and phase artifacts. The new raw training loss is not numerically comparable to the old normalized total.

Final reconstruction goal: 0.99 waveform correlation on active clips, accompanied by level, quiet-audio and perceptual-quality checks. Correlation alone is not perceptual equivalence. Silence is evaluated through absolute noise and level measures rather than unstable near-zero correlations. Full quality scoring and listening establish acceptance after learning; they do not unlock the losses needed for learning.

Before launching, validate exact latent/sample accounting, identity adapter, training-only calibration, frozen teacher, actual GAN/FM gradient flow, checkpoint continuation and batch/stream/fold parity. Stop for numerical failures, corrupted targets or uncertain exposure. Quality gaps are reported rather than disguised as a passed entry gate.

## Sources

The [Supertonic paper](https://arxiv.org/html/2503.23108v3#S3.SS1.SSS2) includes mel reconstruction, adversarial and feature-matching training. The released architecture is useful evidence for the decoder body; it does not supply a complete distillation recipe for AudioVAE2's frozen 25 Hz latents.

[Vocos training code](https://github.com/gemelo-ai/vocos/blob/main/vocos/experiment.py) supports step-based mel pretraining before discriminator training and does not require near-perfect waveform correlation. Its default mel pretraining interval is zero. Our 500-update warmup is an explicit pilot choice, not a published universal requirement.

Training-only discriminators and calibration add no inference-time layers. The new adapter removes its learned input matrices. CPU RTF and final quality remain unmeasured for this newly trained architecture.

## Prepared pilot data

The immutable Runpod plan contains 320,000 optimizer windows from 70,277 recordings, totaling 206.54 scored hours. Duration shares are 31.1% Indic, 31.3% English, 37.1% other languages and 0.56% explicitly labeled expressive events. All 22 scheduled Indian languages and 110 identified speech languages appear in optimization data. Event labels describe recordings, not densely annotated event seconds. There are 512 separate calibration windows from 178 recordings, totaling 20.51 minutes. They have no scored overlap with optimization windows. Original source files may be shared between calibration and optimization.

The dashboard at https://34d6pb4ub5ldrz-8888.proxy.runpod.net/ now reads only the corrected recipe log directory. Historical TensorBoard files and checkpoints are preserved.

## Launch validation

The full-size H100 preflight passed using real frozen teacher targets, all eight discriminators and actual generator feature-matching/adversarial gradients. The maximum streaming-versus-batch sample difference in that check was 5.96e-7. CUDA reflection padding initially blocked deterministic GAN backward; equivalent explicit slicing, reversal and concatenation fixed it with exact CPU value/gradient parity and a passing strict CUDA backward check. Determinism was not disabled. This validates execution, not trained quality.

The 10,000-update training job has started under the single TensorBoard run `decoder-recipe-v2`. The old runs remain saved outside the displayed log directory. No commits or pushes were made for this iteration.

## Monitoring and expanded validation

The current dashboard is generated by a separate observer. It reads saved metrics and evaluations without changing training weights, gradients, normalization or sample order. `loss/reconstruction_only` means waveform plus mel; it is not a complete GAN objective. Individual losses and scheduled versus measured component-gradient norm fractions are shown separately.

The original validation panel stays fixed for learning-curve comparisons. Quiet residual RMS, the repeating 480-sample residual and output peaks remain visible even when strict pass rates are zero. A separate reporting check requires both reconstruction acceptance and peaks within full scale. This check never gates training. Whisper/breathing and missing-language additions have separate appendix results, so they cannot change the historical aggregate.

The observer retains at most two of its milestone checkpoints, with actual step, hash and exposure cursor recorded. Earlier audit snapshots are preserved separately. The live trainer and its original logs are untouched. A retained model snapshot alone does not permit replaying later audio: continuation must preserve the optimizer, global step and complete exposure history.

Additional verified training sources are staged separately from the active immutable plan. Generic emotional nonverbal audio retains that category; it does not count as labeled crying, whistling or another unverified action. New held-out contributors are reserved before training selection. Adding staged data requires a versioned continuation plan, not editing the active sampler in place.
