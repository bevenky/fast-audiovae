# Published methods: what transfers to this decoder

Research checked on 2026-09-10. No experiment or model change was made. The key distinction is between a published audio result and an adaptation to our fixed 64-channel AudioVAE2 latents, strict streaming contract and current failure cases.

## ConvNeXt V2 has audio evidence, but not the claimed isolated remedy

APNet2 is a released ConvNeXt V2 vocoder with separate amplitude and phase predictors. Its ablation replacing ConvNeXt V2 with the original ResNet worsens objective quality. This does not isolate GRN against ConvNeXt V1: the entire backbone changes. Its other ablations replace MRD with MSD, and hinge GAN with least-squares GAN. There is no isolated phase-loss ablation in the reported table. The experiment therefore supports that combined vocoder recipe, not the claim that adding GRN will fix our periodic silence. Its reported single-core CPU RTF is 0.021 versus Vocos 0.009 under its own 22.05 kHz experiment, not our 48 kHz streaming contract. [APNet2 paper](https://arxiv.org/html/2311.11545#S4.SS3)

The official code is not causal. GRN operates on `[B,T,C]` and computes its norm over `T`; depthwise kernel-seven convolutions have symmetric padding-three; synthesis uses `torch.istft(center=True)`. Directly importing this GRN makes earlier samples depend on later frames and on how inference is chunked. A cumulative, fixed-window or framewise substitute would be a new adaptation needing validation. GRN also retains learned additive offsets; it supplies no silence-floor or amplitude-bound guarantee. [Official implementation, GRN/block and generator](https://raw.githubusercontent.com/redmist328/APNet2/main/models.py)

From the normalization formula, GRN is not automatically an unstable inverse-amplitude amplifier: its numerator also shrinks with input amplitude, and the denominator has epsilon. Its central issue here is temporal aggregation and unproven usefulness, rather than a demonstrated numerical defect. It cannot be folded into fixed convolution weights like calibrated BatchNorm because its response depends on the current input.

## The cleanest transferable APNet2 idea costs nothing during decoding

APNet2 applies L1 losses to real and imaginary spectral components in addition to its other objectives. We can evaluate such a paired complex-spectral objective on STFTs of actual student and teacher waveforms while keeping the inference network unchanged. The source uses this alongside, not instead of, other supervision. [Official training code](https://raw.githubusercontent.com/redmist328/APNet2/main/train.py)

This is a redistribution of reconstruction gradients, not new teacher information: waveform supervision already constrains phase. Complex Cartesian differences avoid explicitly dividing by near-zero spectral magnitude. Unweighted wrapped-angle losses can overemphasize unreliable phase near silence; introducing arbitrary inverse-amplitude weights would repeat an earlier training problem. A complex squared-error objective with uniform frequency weights is closely related to windowed waveform squared error by Parseval, so it should not be advertised as fundamentally new supervision.

Do not import APNet2's STFT-consistency term unchanged. That term corrects independently predicted spectral coefficients. Our decoder emits a waveform, whose computed STFT is already consistent by construction. Any useful change would need to be teacher reconstruction supervision, not a redundant consistency loop.

## ComVo: a stronger discriminator mechanism, with measured tradeoffs

ComVo uses genuinely complex-valued discriminator layers, rather than sending real and imaginary channels into an unrestricted real network. Its generator/discriminator ablation keeps MPD and disables phase quantization. Replacing only the discriminator improves MR-STFT from 0.8856 to 0.8679 and PESQ from 3.6266 to 3.6399, but UTMOS falls from 3.6025 to 3.5930 and V/UV F1 from 0.9522 to 0.9497. Thus the training-only transfer has isolated, modest, mixed evidence; the largest gains require the complex generator too. Its 25% training-time result concerns the block-matrix implementation, not a CPU decoder speedup. [ComVo paper, ablation and computation](https://arxiv.org/html/2603.11589#S5.SS3)

The released cMRD removes waveform DC and independently peak-normalizes each clip to 0.8 before extracting its complex spectrogram. That makes the spectral adversary unsuitable as the only supervisor of absolute silence level or peak amplitude. It can greatly rescale tiny quiet residuals. This preprocessing must be treated explicitly if the component is adapted; it is not evidence that the method handles our silence and overshoot automatically. The discriminator is discarded at inference, so adopting only it adds no decoder operations and does not alter inference causality. [Official cMRD source](https://raw.githubusercontent.com/hs-oh-prml/ComVo/main/exp/discriminators.py)

## Which paper directly addresses our periodic mechanism?

Pons et al.'s upsampling analysis is the closest match to the 10 ms repeating residual: subpixel synthesis can create tones because different output positions use different filters. This remains possible without overlapping transposed convolutions. The work studies alternative operators, including nearest interpolation, but interpolation changes frequency response; it is not a demonstrated quality-neutral replacement for our frozen-latent decoder. [Authors' analysis and runnable examples](https://github.com/DolbyLaboratories/neural-upsampling-artifacts-audio/blob/main/ARTICLE.md)

WaveNeXt deliberately adopts direct waveform prediction using a learned linear layer plus reshape, without overlap-add. It reports better synthesis than Vocos while preserving speed in its setup. That is the same broad synthesis family we already use, so faithfully copying its final head does not add a missing periodic-noise safeguard. The paper also describes multi-stream HiFi-GAN's learned, bias-free synthesis filter as a published alternative to fixed subband synthesis. Adapting that mechanism to our head has research precedent, but our channel counts, causal alignment and compute allocation remain engineering hypotheses. [WaveNeXt paper](https://www.okamotocamera.com/preprint_asru_2023_okamoto.pdf)

## Why not simply switch to Vocos or alias-free activation?

The official Vocos head predicts Fourier coefficients and synthesizes them with iSTFT; its released backbone has symmetric temporal convolutions and its overlap-add implementation trims both ends. A strict causal, sample-aligned version therefore needs a changed contract or implementation. It is a useful efficient synthesis precedent, not a directly swappable causal decoder. [Vocos backbone](https://raw.githubusercontent.com/gemelo-ai/vocos/main/vocos/models.py), [synthesis implementation](https://raw.githubusercontent.com/gemelo-ai/vocos/main/vocos/spectral_ops.py)

Aliasing-Free Neural Audio Synthesis supplies stronger evidence about nonlinear aliasing and tonal artifacts, with ADAA SnakeBeta and filtered resampling. However, its latest version explicitly reports higher CPU cost from filtering/oversampling and targets offline production. It does not establish strict causality. Its test-signal results and general audio experiments are relevant to whistling-like tonal material, but do not demonstrate that our already measured low-level residual is caused by nonlinear aliasing. [Latest paper](https://arxiv.org/html/2512.20211v3)

## Recommendation from this literature review

Do not replace ConvNeXt with ConvNeXt V2 merely because it is newer. The published implementation adds temporal dependence incompatible with our streaming contract, and its audio ablation does not isolate GRN's value.

The most conservative published-method experiment is paired complex-spectral reconstruction on the existing decoder, with the current nonlinear head jointly trained and an equal-training control. A true complex discriminator is a subsequent, separately controlled training candidate; its quiet-level normalization deserves particular care. These are zero-inference-cost changes, but neither is a demonstrated solution and neither replaces safe optimizer updates.

If architecture changes remain necessary, learned multi-stream synthesis has published precedent and a direct connection to the teacher's final multichannel synthesis. Prefer stating that mechanism and its capacity/CPU tradeoffs over presenting the proposed 32-channel intermediate head as a published architecture or a guaranteed fidelity improvement. Output bounding remains separate: a jointly trained tanh guarantees range but cannot establish quiet accuracy or faithful transient shape.

No reviewed paper establishes all of our desired properties together: unchanged AudioVAE2 latents, teacher-level reconstruction across our edge cases, strict chunk-invariant causality and no additional single-core CPU time.
