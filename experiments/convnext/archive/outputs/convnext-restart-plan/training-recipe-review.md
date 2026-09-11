# Training recipe for the decoder restart

Research and proposed design, 9 September 2026. No training, inference, benchmarks or training-source changes were performed for this review. The target remains the original frozen AudioVAE2 encoder and decoder. Supertonic provides architectural and training ideas; its weights are not needed.

## What the sources establish

The [Supertonic paper, version 3](https://arxiv.org/html/2503.23108v3#B1.SS1), predates Supertonic 3. Its autoencoder recipe is:

| Component | Published specification |
|---|---|
| Generator | Mel reconstruction, adversarial, feature matching; coefficients 45, 1, 0.1 |
| Reconstruction | FFT 1024/2048/4096; mel bands 64/128/128; Hann, window=FFT, hop=FFT/4 |
| Discriminator labels | Least squares: real +1, fake −1; generator targets +1 |
| Feature matching | Layer-average L1 distance |
| MPD | Periods 2/3/5/7/11; channels 16/64/256/512/512/1 |
| MRD | Log-linear spectra; FFT 512/1024/2048, quarter hops, Hann; five 16-channel layers then one-channel output |
| MRD kernels/strides | Kernels 5/5/5/5/5/3; strides (1,1)/(2,1)/(2,1)/(2,1)/(1,1)/(1,1) |
| Training | AdamW, 0.0002, batch 128, 1.5M updates, four RTX 4090s; adversarial crops 0.19 s |
| Decoder normalization | BatchNorm after stem and after ten blocks |

Unspecified: reconstruction log/linear scaling, mel convention, tensor reductions, discriminator normalization, optimizer betas/decay, decoder learning-rate schedule and reconstruction-only warmup. Its separate text-to-latent schedule must not be assigned to the autoencoder. [Architecture appendix](https://arxiv.org/html/2503.23108v3#A1.SS1.SSS2).

The current [official GitHub tree](https://github.com/supertone-inc/supertonic/tree/7e2804f96016a7028cb1ed627353c61c1e9dd281) and [official model tree](https://huggingface.co/Supertone/supertonic-3/tree/3cadd1ee6394adea1bd021217a0e650ede09a323) expose inference implementations and ONNX assets, not the autoencoder training source, encoder weights or discriminator checkpoints. Consequently, Supertonic 3's exact training recipe cannot be recovered from these releases. Its [configuration](https://huggingface.co/Supertone/supertonic-3/blob/3cadd1ee6394adea1bd021217a0e650ede09a323/onnx/tts.json) gives 44.1 kHz, 24-channel latents, ten 512/2048 decoder blocks, and a 512→2048→512 head. It also declares an encoder input width of 1253 despite 228 mel bands in its spectrogram configuration; those metadata do not establish a complete executable encoder.

The following is our declared student recipe, including the choices the sources leave open. It is not a claim to reproduce unpublished Supertonic 3 training.

## Correct the target and reconstruction objective

For each original utterance, let `z` be the frozen teacher's raw 64-channel posterior mean, `t = teacher_decoder(z)` at 48 kHz, and `p = student(z)`. Every training branch initially compares `p` with `t`. Preserve the original waveform for independent evaluation, including its actual bandwidth; do not impose the original 16 kHz signal as a competing full-band training target.

Use two reconstruction losses:

1. **Teacher mel L1.** At 48 kHz, use FFT/window sizes 1024/2048/4096 and hops 256/512/1024; periodic Hann, unnormalized STFT magnitude, `power=1`, one-sided spectrum. Apply actual triangular mel filters with 64/128/128 bands, `mel_scale=slaney`, area-normalized filters, 0–24 kHz. Use the sum of linear magnitude-mel L1 and `log(clamp(mel, min=1e-5))` L1. Average each absolute error within its spectrogram, then equally across resolutions and examples. Use `center=False` on the actual scored slice. These scaling, filter and reduction choices are explicitly ours. The [official Vocos loss](https://github.com/gemelo-ai/vocos/blob/eb39abfc42c1dee4854d9b10d44dd7d4fd3b0e53/vocos/loss.py#L9) provides an established log-mel magnitude L1 implementation, but uses centered STFT and is not identical to this proposal.
2. **Energy-normalized teacher waveform L1.** For each example, divide `mean(abs(p-t))` by `max(rms(t), 0.001)`, with the denominator detached. Average examples. The floor bounds quiet/silent-target amplification. The same target-derived scale applies to the error; never normalize student and teacher independently, which would conceal amplitude mistakes. Also log unnormalized L1, RMS ratio, lag, cosine and SNR.

Loss windows must exclude left context and padded tails before STFT or discriminator input. Retain the existing minimum scored length. Neither zeros nor reflected content outside the scored region should be counted as newly observed audio. Exact teacher output must produce zero reconstruction error to numerical tolerance. Silence, scaled-teacher and shifted-teacher controls must expose energy and alignment errors.

## Make gradient influence explicit

Use an EnCodec-style output-gradient balancer, not the old 45/15/1 coefficients transplanted onto new losses. The [official EnCodec implementation](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/balancer.py#L73) differentiates each loss at generated audio, estimates per-example gradient norms, maintains an EMA, and rescales each component before one network backward pass. Its ratios describe desired output-gradient contributions, not guaranteed parameter updates.

Proposed initial settings: reconstruction gradient shares 50% mel and 50% waveform; EMA decay 0.999, epsilon 1e-12, reference norm 1. The shared output and valid-sample masks must be identical across branches. Treat the first nonzero measurements as initialization, checkpoint the EMA accumulators, and bound rescaling factors to [0.0001, 10000]. Zero-gradient terms remain zero and are excluded from saturation alarms. Those bounds and shares are our proposed safeguards, not published Supertonic settings. Log effective scales, individual/combined norms and saturation; persistent clipping of nonzero terms is a failed calibration gate, not a condition to hide.

At preflight and selected checkpoints, inspect component gradients in the adapter, a middle block and output head. Output-space balance alone does not establish useful parameter-space influence or successful learning. Keep parameter clipping and finite checks. No balancer or discriminator enters the deployed decoder.

## Implement the perceptual stage before a long run

Build the lightweight MPD and MRD described above. For underspecified MPD details, declare the conventional kernel-5/stride-3 stack, fifth-layer stride 1, kernel-3 output, LeakyReLU 0.1 and weight normalization as choices informed by [HiFi-GAN's official implementation](https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/models.py#L128). Use weight normalization and LeakyReLU 0.1 for MRD too, explicitly as our choices. Do not silently substitute Vocos's complex, multiband discriminator, which also normalizes its input volume.

Use raw teacher output as the discriminator's real waveform. Do not independently peak-normalize real and generated inputs. Select the same contiguous, randomly positioned 9120-sample (0.19 s) crop from each scored pair; use the entire valid crop when shorter. Retain full scored windows for reconstruction. For short valid MRD inputs, the minimum existing 4096-sample policy accommodates all three FFTs.

Declare all reductions:

- `L_D = mean_k [mean((D_k(t)-1)^2) + mean((D_k(stopgrad(p))+1)^2)]`.
- `L_adv = mean_k mean((D_k(p)-1)^2)`.
- `L_fm = mean_(k,l) mean(abs(phi_(k,l)(p)-stopgrad(phi_(k,l)(t))))` across the five hidden feature maps of each discriminator; exclude final logits.

Here `k` spans all eight discriminator heads. During a generator update freeze discriminator parameters while retaining gradients through their input. During a discriminator update detach the student waveform. Use one discriminator update per generator update after activation, and checkpoint both states and the phase/crop RNG.

Proposed full-stage output-gradient shares are 40% mel, 30% waveform, 20% feature matching and 10% adversarial. Ramp the latter two from zero over 500 generator updates while reducing mel from 50% to 40% and waveform from 50% to 30%. This is our proposed transition, not a documented Supertonic schedule. Do not add unimplemented perceptual terms merely to the plan or dashboard.

Keep the existing Muon/AdamW generator split to isolate the identified objective failure. Use AdamW for discriminators. Declare discriminator learning rate 0.0002, betas 0.8/0.9 and zero weight decay as an initial choice; Vocos's current official [training code](https://github.com/gemelo-ai/vocos/blob/eb39abfc42c1dee4854d9b10d44dd7d4fd3b0e53/vocos/experiment.py#L74) supports AdamW 0.8/0.9 and cosine scheduling, but does not establish Supertonic's settings. Schedule the bounded run's learning-rate decay explicitly after warmup, with its endpoint fixed before launch. No evidence currently warrants abandoning Muon or promising that an optimizer change fixes the audio.

## Gates before consuming the larger corpus

| Gate | Required evidence |
|---|---|
| Loss and checkpoint tests | Exact-teacher zero reconstruction; amplitude/shift controls; gradient routes; no teacher updates; no padding contribution; deterministic resume including balancer, discriminators, schedule and sampler |
| Full-capacity tiny-set overfit | A small fixed, explicitly designated debugging set containing speech, whisper and whistle. Require waveform recovery, not just spectral-total reduction. Proposed engineering targets: at least 90% relative waveform-error reduction, RMS ratio 0.95–1.05, cosine above 0.95 and SNR above 20 dB on each non-silent training example. Cap the exercise; investigate if it fails. These are gates, not predicted results. Repeated debugging examples must be declared separately from unique large-corpus exposure. |
| Perceptual integration | A bounded real-audio trial with active discriminator and feature losses; finite useful generator/discriminator gradients; audible speech and expressive content; no new gain collapse or temporal offsets |
| Fixed development panel | Paired teacher/student audio, raw energy and phase checks, existing speech metrics on valid speech and explicit listening coverage for non-speech. Keep one panel fixed across checkpoints. MOS predictors alone cannot validate whistles or phase fidelity. |
| Streaming and normalization | Evaluate at fixed statistics, with matching batch/chunk output and exact sample accounting. Any two-BatchNorm variant must pass variable-length padding-statistics checks and export/folding parity. BatchNorm is a bounded architectural candidate supported by the sources, not a proven cause of the failed run. |

Longer training starts only after these gates, with the perceptual stage already runnable. A reconstruction-only pass cannot be described as completing the high-quality decoder recipe. GAN totals are not expected to approach zero, so dashboard decisions must use component losses and fixed audio diagnostics rather than a threshold such as 21 or 22.

## Source pins and limitations

| Source | Revision inspected |
|---|---|
| Supertonic paper | arXiv 2503.23108v3, 23 September 2025 |
| Official main repository | `7e2804f96016a7028cb1ed627353c61c1e9dd281` |
| Official Python SDK | `908a56486e821e833a80530ff0cae3ad0b046fce` |
| Supertonic 3 model assets/config | `3cadd1ee6394adea1bd021217a0e650ede09a323` |
| EnCodec reference balancer | `0e2d0aed29362c8e8f52494baf3e6f99056b214f` |
| HiFi-GAN discriminator reference | `4769534d45265d52a904b850da5a622601885777` |
| Vocos loss/training references | `eb39abfc42c1dee4854d9b10d44dd7d4fd3b0e53` |

An [unofficial reproduction](https://github.com/ORI-Muchim/supertonic-training/tree/24c5d8e802e7bcf6f81601ffeae6886d70fa98c6) was also inspected. It explicitly chooses mean reductions and log-mel loss and reports a single-speaker KSS run. It is useful for comparing implementation choices, but is not Supertone's trainer, does not establish Supertonic 3's recipe and does not validate our teacher-conditioned student. No unofficial source or model weights are required for the proposal.
