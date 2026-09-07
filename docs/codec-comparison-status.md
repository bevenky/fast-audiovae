# FLEURS codec quality comparison

These are the existing measured scores from **60 FLEURS clips across 10 languages**, six clips per language: six Indic languages, English, and three Romance languages. No new quality scoring was performed in the current Intel optimization iteration. See the [original quality report](multilingual.md) and [saved results](../benchmarks/multilingual/quality-summary.json).

| Codec | PESQ-WB ↑ | STOI ↑ | UTMOS22 strong ↑ | DNSMOS P.835 OVRL ↑ | DNSMOS P.808 ↑ | MUSHRA |
|---|---:|---:|---:|---:|---:|---|
| Fast AudioVAE2 | 3.7415 | 0.9360 | 2.2569 | 2.7653 | 3.4039 | Unmeasured |
| Mimi, Pocket continuous checkpoint | 2.1300 | 0.8074 | 2.5168 | 2.8940 | 3.3387 | Unmeasured |
| Meta DACVAE | 4.2838 | 0.9731 | 2.2217 | 2.7790 | 3.4313 | Unmeasured |
| Supertonic 3 | N/A | N/A | N/A | N/A | N/A | Unmeasured |

Each numeric cell has 60 valid measurements, with no recorded metric errors. Means give equal weight to languages, which also equals the clip mean here. Scoring ran on the Apple CPU. The stock AudioVAE2 control matches Fast AudioVAE2 at the displayed precision.

Meta DACVAE leads PESQ, STOI, and DNSMOS P.808 on this sample. Mimi leads UTMOS22 and DNSMOS P.835 OVRL. The MOS predictors estimate different properties from reference-based reconstruction metrics; their ranking is not a human listening result or proof of better source fidelity. UTMOS22 here is the single strong learner, not UTMOSv2. No controlled MUSHRA listening study has been performed for any row.

Supertonic 3 remains unmeasured because its [official release](https://huggingface.co/Supertone/supertonic-3/tree/main/onnx) provides a vocoder but no matching audio encoder checkpoint for waveform reconstruction. Its text encoder cannot supply that function. TTS outputs, unrelated codecs, and reconstructed encoders would not establish the missing stock Supertonic 3 scores.

The corpus and scoring waveforms are 16 kHz, so these results do not establish fidelity above 8 kHz. The checkpoints also differ in latent capacity, output rate, and causality: Meta DACVAE is noncausal, while the measured AudioVAE2 and Mimi paths are causal. Intel numerical waveform-parity checks are separate validation; they do not mean that MOS scoring was rerun on the latest Intel kernels.
