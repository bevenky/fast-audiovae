# Supertonic 3 decoder comparison

Inspected directly on 2026-09-09. This is a static graph and source audit, with no inference, training, timing, weight conversion or imported Supertonic parameters.

**Our student includes the released decoder's ten ConvNeXt blocks and its full nonlinear direct-waveform head. There is no missing Snake, iSTFT, progressive upsampling network or ConvNeXt V2 GRN stage. The meaningful differences are the AudioVAE2 input adapter, output rate/block size, and normalization during training.**

The local `vocoder.onnx` is 101,424,195 bytes with SHA-256 `085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba`. Fresh official Hugging Face metadata confirms that exact file remains current at revision `3cadd1ee6394adea1bd021217a0e650ede09a323`. The graph was produced by PyTorch 2.9.0, ONNX opset 19. It contains 401 nodes: 33 convolutions, ten LayerNorm operations, ten Erf-based GELUs, one final BatchNorm and one shared PReLU. [Official graph and checksum](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/vocoder.onnx), [official configuration](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/tts.json).

## Decoder stages

Graph indices are zero-based and appear in the accompanying JSON. Student references are to `experiments/convnext/audiovae_student/model.py` in the repository.

| Stage | Released Supertonic 3 graph | Current student | Assessment |
| --- | --- | --- | --- |
| Input contract | `[B,144,L]`, representing 24 channels packed across six frames | Raw AudioVAE2 means `[B,64,T]` at 25 Hz | Required interface change |
| TTS unpacking and normalization | Divide by 0.25; reshape `[B,24,6,L]`; transpose into chronological order; unpack to `[B,24,6L]`; multiply 24-channel std and add mean, nodes 0–34 | New biased 1×1 adapter 64→256, four chronological phase frames per latent, lines 117 and 133–136 | Intentional adaptation. Supertonic's statistics belong to its own encoder/TTS latent space and are not appropriate values for raw AudioVAE2 means |
| Stem | Biased 24→512 convolution, kernel 7, stride 1; node 51 | Biased 64→512 convolution, kernel 7, stride 1; line 118 | Same body width and temporal kernel; changed input channels |
| Causal padding | Edge replication only on the left: 6 samples for stem; 6 × dilation for blocks; 2 for head | Per-layer left replication at real starts; explicit history for streaming, lines 59–73 | Preserved |
| Temporal processing | Ten depthwise 512-channel kernel 7 convolutions, groups 512 | Ten matching depthwise convolutions, lines 76–92 and 119 | Preserved |
| Dilation schedule | `[1,2,4,1,2,4,1,1,1,1]` | Same schedule, line 22 | Preserved in frame units |
| Per-block normalization | Transpose to `[B,T,C]`; LayerNorm on channels only, epsilon 1e-6, learned scale/bias | Same LayerNorm axes and epsilon, lines 81 and 87 | Preserved |
| Per-block channel mixing | Biased 1×1 convolutions 512→2048→512 | Biased Linear 512→2048→512 on the channel-last view, lines 82–88 | Same mathematical pointwise transformation; different operator spelling |
| Per-block activation | Exact `0.5*x*(1+erf(x/sqrt(2)))` GELU | Exact GELU, `approximate='none'`, line 88 | Preserved |
| Per-block residual | Learned gamma shape `[512,1]`, then residual addition | Learned 512-channel scale, then residual addition, lines 84 and 89 | Same operation; our 1e-6 initialization is a proposal, not a recovered Supertonic training setting |
| Final normalization | BatchNormalization with learned affine values and fixed running statistics, node 372; epsilon 1e-5, training_mode 0 | Learned channel affine initialized to identity, lines 95–102 and 120 | Same inference function class, different training behavior; see below |
| Nonlinear waveform head | Biased 512→2048 convolution, kernel 3, node 389 | Matching biased 512→2048 kernel 3 convolution, line 121 | Preserved |
| Head activation | One scalar PReLU slope, exported shape `[1,1]`, node 390 | One shared PReLU slope, line 122 | Preserved; our 0.25 initialization is not verified as Supertonic's initial setting |
| Waveform projection | Bias-free 2048→512 pointwise convolution, node 391 | Bias-free 2048→480 pointwise convolution, line 123 | Required output-rate/block-size adaptation |
| Waveform assembly | Transpose `[B,512,T]` to `[B,T,512]`, flatten chronologically; nodes 392–400 | Matching chronological flatten with 480 samples per internal frame, lines 138–147 | Preserved mechanism |
| Terminal processing | No tanh, clamp, iSTFT or further refinement | None | Preserved |

Both bodies use 116 internal history frames. At Supertonic's 44.1 kHz/512 clock this is about 1.347 seconds; at our 100 Hz clock it is 1.16 seconds. This is an intentional physical-time difference despite matching dilation numbers. It is still longer than the original AudioVAE2 decoder's required latent history, according to the separate alignment audit. The student still receives one new latent per 40 ms; four internal phases do not make its external codec interface 10 ms.

## The normalization difference needs an explicit decision

Supertonic's node 372 is not an inferred label: it is an actual inference `BatchNormalization` operation with four 512-element parameter/statistic arrays. At inference its equation is `a*x+b`, where `a = gamma/sqrt(running_var+epsilon)` and `b = beta-a*running_mean`. A channel affine can represent exactly that same function class without copying any learned numbers.

That does **not** establish training equivalence. BatchNorm may have standardized hidden activations using batch/time statistics during training, improving or altering conditioning of the waveform head. Our affine does not standardize current activations; it learns only a scale and offset. The exported graph identifies fixed inference statistics but does not reveal whether the training implementation used BatchNorm, SyncBatchNorm, frozen statistics, unusual schedules, or different handling of padded examples.

The stem is another export uncertainty: its weight and bias have anonymous exported names `onnx::Conv_1441/1442`. The paper's appendix A.1.2 explicitly describes BatchNorm after the stem and after the full block stack. That makes folded stem normalization a supported interpretation, but a biased V3 graph stem alone cannot recover its training implementation. The two-site normalization change is paper-motivated; adding BatchNorm after every individual ConvNeXt block would be a different proposal. [Paper decoder description](https://arxiv.org/html/2503.23108v3#A1.SS1.SSS2).

A restart plan should label normalization as a deliberate experimental difference and decide how to condition activations using our own data. Adding train-mode BatchNorm blindly could mix padded frames and other examples into statistics, while strict streaming inference must remain causal. No direct reuse of Supertonic statistics is needed or proposed.

## Adapter capacity and exact folding requirements

The 64→256 pointwise phase adapter does not impose another dimensional bottleneck. Before the permutation, its output is a 256-dimensional affine map of the 64-dimensional input. Each of its four 64-channel phases can have a full-rank 64×64 map, so each phase can retain every input coordinate. The permutation discards nothing. This is a statement about available capacity, not a measurement of the trained matrices' conditioning or rank. Four phases do not create additional source information; the nonlinear body must learn how to render the same 40 ms latent across four 10 ms output phases.

The adapter and the new latent distribution remain unvalidated inductive biases. However, there is no demonstrated information-theoretic reason they must lose fidelity beyond AudioVAE2's encoder. A fixed, invertible per-channel normalization estimated from our own training latents is an alternative conditioning change that preserves raw-mu API semantics and can fold into the adapter. It needs no borrowed Supertonic statistics or sample-dependent inference operation.

Our 100 Hz body evaluates blocks `100 / (44100/512) = 1.160998` times as often as the released 86.1328 Hz body. The head emits 480 rather than 512 samples. Matching block counts and widths therefore does not guarantee exactly Supertonic's decoder RTF. No new timings were run for this audit.

For a fixed normalized channel `y = a*x+b`, folding can preserve the current causal inference function:

- **Normalization after the stem convolution:** multiply each output-channel kernel by its `a`, then replace the stem bias with `a*bias+b`.
- **Normalization before the head convolution:** multiply the kernel on its input-channel axis by `a`; add `sum(input_channel,kernel_position, original_kernel*b)` to the head bias.
- **Why replicate padding matters:** `replicate(a*x+b) = a*replicate(x)+b`, including startup padding. That identity makes the pre-head fold valid at the beginning of an utterance as well as steady state. It would generally fail at a zero-padded boundary without an explicit correction.
- **Streaming state:** start new streams after folding/export. The current head history stores inputs after the affine. A folded head consumes the corresponding unnormalized history; reusing an old live state would mix representations. Validating fresh-state whole/stream parity is required.

Masked BatchNorm must exclude artificial right padding from its statistics. A policy must also specify whether real but unscored left context and partially valid final output phases contribute; repeated context is not new training audio. Running statistics should be calibrated using training-only material and fixed weights, then frozen. If two sites are used, calibrate the stem first and measure the final site under that frozen stem, or otherwise ensure the final statistics correspond to the final upstream function. Do not calibrate on held-out dev clips.

Even correctly masked train-mode BatchNorm uses batch/time statistics and therefore future frames during training. Fixed-stat inference is causal, but the training/inference behavior differs. If causality must hold in every training forward as well, use fixed statistics from the outset or a per-frame normalization. Per-frame LayerNorm/RMSNorm is causal and avoids running-stat calibration, but it is input-dependent, changes the function class, and cannot fold into a static convolution. These are tradeoffs to decide before a restart, not evidence that BatchNorm caused the present failure.

## STFT and iSTFT are different roles

An **STFT training loss** transforms an already generated waveform into a spectrum, compares it with a target spectrum, and backpropagates an error. It is used only while training. It does not put a Fourier transform into the inference decoder.

An **iSTFT synthesis head** predicts spectral coefficients and uses an inverse transform, usually with overlap-add, to construct the output waveform. That is an inference architecture choice. The inspected Supertonic 3 graph ends with a learned waveform projection and reshape; it does not use this choice. Our current decoder also projects directly to waveform samples. Our `losses.py:45–53` uses STFT only to score those samples.

The official configuration also describes an **encoder** spectrum processor with FFT 2048, hop 512 and 228 mel bands. That is another use of STFT, on the input side of Supertonic's own audio autoencoder. It does not imply an iSTFT decoder. We have intentionally retained AudioVAE2's existing encoder instead. The current official Hugging Face asset list includes the TTS text encoder, duration predictor, vector estimator and vocoder, but no audio-autoencoder encoder checkpoint. [Configuration](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/tts.json), [official assets](https://huggingface.co/Supertone/supertonic-3/tree/main/onnx).

The `convnext_2` configuration key is inside the TTS vector-field network and names one indexed block group. It is not evidence that the audio decoder implements ConvNeXt V2. The decoder graph contains ordinary LayerNorm/GELU/channel-scale blocks, without dynamic global-response normalization.

## What the release cannot establish

The official GitHub repository supplies ONNX inference integrations; its Python helper invokes the vocoder graph. It is not a full release of the V3 audio-autoencoder training program. The exported vocoder does not identify discriminator architecture, losses, optimizer, initialization, training duration, stochastic depth/dropout used only during training, or the encoder checkpoint that produced its latent distribution. Those require paper/source evidence and must not be silently inferred from successful inference. [Official repository](https://github.com/supertone-inc/supertonic), [official Python inference helper](https://github.com/supertone-inc/supertonic/blob/main/py/helper.py).

The architectural audit does not support adding a supposedly missing Snake, iSTFT, GRN or upsampling stage. It supports retaining the complete lightweight body/head while addressing the measured objective imbalance, the deliberate normalization difference, the new low-rate latent adapter, and the missing perceptual training curriculum before deciding whether the architecture itself needs replacement.

Detailed evidence: `supertonic3-static-graph-audit.json` records all 401 graph nodes, initializer shapes only, exact padding constants, norm attributes and current release identity. No learned weight, mean/std or running-statistic values are exported in that audit. The only released scalar value included is the TTS wrapper's configured 0.25 scaling factor.
