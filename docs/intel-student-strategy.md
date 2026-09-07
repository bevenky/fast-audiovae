# A faster codec student

**Prioritize a student pilot after the accepted Intel checkpoint.** The completed Intel campaign passed all 264 validation records and improved decoder RTF from 0.25398 to 0.24424. Retain it as the validated baseline. A Supertonic-style decoder offers a larger potential reduction in computation than further optimization of the same AudioVAE2 network. Matching the audio quality you prefer is still an unproven training objective.

The architectural difference matters. AudioVAE2 processes residual blocks at progressively higher rates, reaching 48,000 steps per second. Supertonic keeps its main network at a low frame rate, then predicts waveform blocks directly. Its released decoder uses causal ConvNeXt-style blocks, GELU and a PReLU waveform head. It does not use Snake or ConvNeXt V2's GRN. [Supertonic architecture](https://arxiv.org/html/2503.23108v3#S3.SS1)

Our static count illustrates the opportunity:

| Decoder | Matrix/convolution GMAC per audio second |
|---|---:|
| Existing AudioVAE2 | 8.9912432 |
| Hypothetical Supertonic-style decoder at 48 kHz, 512-sample hop | 2.37072 |

That is approximately **74% fewer matrix/convolution operations**, not a predicted 74% reduction in RTF. The count excludes activation functions, normalization, memory traffic, packing and encoder computation. It assumes the same decoder widths and depth as the released Supertonic topology, retrained at 48 kHz. [Static counts and assumptions](../benchmarks/intel-upsampling/architecture-costs.json)

The proposed codec should have a **native 48 kHz input, native 48 kHz output, causal encoder and causal decoder**. Supertonic's released model outputs 44.1 kHz, so this would be a newly trained architecture, not a sample-rate setting change. Its public assets do not supply a complete audio encoder and training implementation. [Official configuration and assets](https://huggingface.co/Supertone/supertonic-3/tree/main/onnx)

The latent interfaces also differ: AudioVAE2 uses 64 dimensions at 25 frames/s, or 1,600 floating-point values/s. The proposed 24-dimensional representation at 93.75 frames/s would use 2,250 values/s. It is neither a compatible latent replacement nor a matched compression budget. These are representation counts, not coded bitrates.

Use **original fullband audio as the primary reconstruction target**, with DAC-VAE supplying additional waveform or feature supervision during training only. The teacher's runtime cost would disappear from deployment. Its noncausal analysis path must not become the deployed encoder, and a causal student should not be required to exactly match future-dependent teacher latents. [Meta DAC-VAE implementation](https://github.com/facebookresearch/dacvae/blob/main/dacvae/model/dacvae.py)

The pilot should establish reconstruction quality on held-out native fullband multilingual audio, including listening comparisons, then measure complete encoder and decoder CPU performance and streaming behavior. Changing architecture cannot inherit the numerical-parity evidence from our kernel work. Advance only if quality holds and the measured gain is worthwhile. **This student has not been implemented, trained or benchmarked.**

## Data and training budget

For a new native 48 kHz causal encoder and decoder, budget **300 to 500 hours for a serious multilingual pilot**, followed by **1,000 to 3,000 hours only if held-out learning curves justify scaling**. These are planning estimates, not published minimums or guarantees of teacher quality. Use speaker-disjoint validation and genuine fullband recordings. Our 16 kHz FLEURS material remains useful for regression testing but cannot establish quality above its source bandwidth.

A decoder-only experiment using an existing encoder could start with around 100 hours to test feasibility. That is a different scope: retaining AudioVAE2's existing encoder preserves its 16 kHz input interface, while retaining Meta DAC-VAE's analysis path would not provide our required causal codec.

The earlier Supertonic paper trained its complete speech autoencoder on **11,167 hours**, about 14,000 speakers, for 1.5 million updates on four RTX 4090s. It describes the audio encoder in Appendix A.1.1: a 228-mel input, 512-channel stem, ten ConvNeXt blocks with 2048-channel expansion, and a 24-channel projection. Only the decoder is explicitly described as causal. This is not the complete Supertonic 3 recipe; the released v3 configuration instead specifies 1253 encoder input features. [Earlier paper](https://arxiv.org/html/2503.23108v3), [v3 configuration](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/tts.json)

No verified source establishes our one-H100 or eight-H100 training time. A duration estimate needs actual step throughput, effective batch, crop duration, teacher caching and the number of updates required for convergence. Dividing another model's training time by GPU count would not establish that.

## Most useful newer work

| Candidate | What is useful | Limitation for this project |
|---|---|---|
| [BlueCodec](https://github.com/maxmelichov/blue-codec) | Released 44.1 to 44.1 kHz continuous encoder/decoder weights and autoencoder training code; very similar ConvNeXt/direct waveform design | Its encoder uses symmetric padding and is noncausal. It is a separately trained model, not Supertonic 3's missing encoder. |
| [FocalCodec-Stream](https://github.com/lucadellalib/focalcodec) | Released streaming codec weights and an explicit stateful WaveNeXt decoder | 16 kHz input to 24 kHz output, with bounded encoder lookahead. The complete streaming training recipe was not verified. |
| [StreamCodec2](https://arxiv.org/html/2509.13670v1) | Closest paper to the proposed noncausal-teacher to smaller causal ConvNeXt V2 student method | The experiment is at 16 kHz; no matching author implementation/checkpoint was found. Its distillation improves PESQ from 2.650 to 2.744, still below the teacher's 3.132. |
| [WaveNeXt 2](https://arxiv.org/html/2605.25506v1) | Newer ConvNeXt vocoder with iterative residual refinement | Adds inference passes and lacks a paired learned encoder. Its paper's single-core CPU RTF is 0.10 to 0.24 for GAN variants, versus 0.06 for original one-pass WaveNeXt, on the same AMD EPYC test setup. |

**Use StreamCodec2 as training-method evidence, BlueCodec as autoencoder scaffolding, and FocalCodec's decoder for streaming implementation ideas.** Start with a single-pass causal waveform decoder. None of these sources proves that our proposed student will match DAC-VAE quality or beat Mimi's CPU RTF.
