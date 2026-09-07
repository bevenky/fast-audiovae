# A faster codec student

**Prioritize a student pilot after the accepted Intel checkpoint.** Finish the current Intel campaign and retain its validated implementation as the baseline. A Supertonic-style decoder offers a larger potential reduction in computation than further optimization of the same AudioVAE2 network. Matching the audio quality you prefer is still an unproven training objective.

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
