# Supertonic 3 source audit

Read-only review, 2026-09-10. No model execution, weight download, training, benchmark or runner changes.

**There is no evidence that Supertonic 3 contains an omitted, special silence or peak-repair layer. Its efficient graph supports the architecture choice; its published results do not establish AudioVAE2-equivalent codec reconstruction under our frozen-latent and edge-case contract.**

## What was verified against the current release

The current Hugging Face [vocoder file page](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/vocoder.onnx) displays SHA256 `085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba` in its Xet pointer details, browser-extracted lines 124-125. This exactly matches the existing local graph audit. The Xet object hash is a different field and was not substituted for the SHA256. No weight file was fetched. The file page identifies the initial release commit as `724fb5a`.

Local evidence: `outputs/convnext-restart-plan/supertonic3-static-graph-audit.json`, produced by `work/convnext-preflight/audit_supertonic3_graph.py`, and `outputs/convnext-restart-plan/supertonic3-decoder-comparison.md`. The comparison is a historical architecture audit, not a claim that its descriptions of our then-current normalization remain the current training state.

| Point | Verified evidence | Consequence for this project |
| --- | --- | --- |
| Latent contract | Released wrapper accepts 144 channels, unpacks six chronological 24-channel frames, reverses the configured 0.25 scaling and applies its own channel mean/std. | These are Supertonic's latent coordinates and statistics. They cannot be copied onto AudioVAE2 posterior means. |
| Decoder clock | 512 waveform samples per internal frame at 44,100 Hz, approximately 86.13 frames/second. | Our four internal phases per 25 Hz latent run at 100 Hz and emit 480 samples each at 48 kHz. Identical block counts do not imply identical RTF; our body runs approximately 16.1% more often. |
| Body | Stem 24→512, kernel 7; ten ConvNeXt blocks, width 512, expansion 2048; depthwise kernel 7, dilations 1,2,4,1,2,4,1,1,1,1; channel LayerNorm, GELU, learned residual scale. | This is already the principal architecture family in our student. |
| Head | Fixed inference BatchNorm; causal kernel-3 convolution 512→2048; one shared PReLU; bias-free pointwise projection 2048→512; chronological flattening. | The nonlinearity is PReLU at the head, not Snake. The blocks use GELU. |
| Absent operations | No Snake/Sin, iSTFT, progressive waveform upsampling network, ConvNeXt V2 dynamic GRN, final tanh or clipping in the released graph. | Adding these cannot be described as restoring missing Supertonic 3 stages. |
| Causality | Every inspected temporal convolution has left-only replicated padding; channel LayerNorm and fixed inference BatchNorm do not use future frames. History is 116 internal frames. | The decoder is causal in its internal frame sequence. Its roughly 1.347 seconds of past context is not lookahead. Packed input groups and the unavailable acoustic encoder prevent asserting a complete streaming codec latency from this graph alone. |
| Normalization | Final BatchNorm is explicit, with inference mode and fixed statistics. A folded stem BatchNorm is plausible but not recoverable solely from the anonymous biased stem convolution. | Fixed affine normalization can be folded into adjacent convolutions, retaining replicated-boundary behavior. Exact training statistics, masking and calibration schedules are not established by the graph. |

The public [configuration](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/tts.json) confirms the audio rate, dimensions and dilation schedule. Its `convnext_2` key names an indexed group in the text-to-latent vector-field network, not ConvNeXt V2 in the decoder.

## Published recipe versus released version

The linked paper is **SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System**, v3 dated September 23, 2025. WaveNeXt is a separately cited architecture influence. The paper trains an acoustic encoder and decoder together before text-to-latent training. Its encoder consumes 228-band log-mel features and produces 24-dimensional latents. Decoder normalization is described after the stem and block stack. Generator training combines multi-resolution mel L1, least-squares adversarial training with MPD/MRD, and discriminator feature matching, weighted 45/1/0.1. It reports 11,167 audio hours, 1.5 million autoencoder updates, batch 128, AdamW and four RTX 4090s. Its reconstruction evaluation uses NISQA, UTMOSv2 and voiced/unvoiced F1; the reported 0.0006 RTF is on an RTX 4090. Neither exact waveform correlation nor our quiet/transient panel is established. These are the **paper's recipe and results**, not a released V3 training implementation. [Paper, sections 3-5 and appendices A-B](https://arxiv.org/html/2503.23108v3).

The current V3 configuration's encoder `idim` is **1253**, not 228 or 275, although a 228-mel spectrum processor is listed. This difference prevents treating the paper's frontend as a verified description of the released V3 frontend. Its full acoustic encoder implementation, checkpoint and future-context behavior were not found in the inspected public release; no factorization of 1253 is inferred here. The [asset tree](https://huggingface.co/Supertone/supertonic-3/tree/main/onnx) contains duration predictor, **text** encoder, vector estimator and vocoder ONNX models. The text encoder is not an audio encoder. The configuration's `n_delay=0` is insufficient to prove causal end-to-end encoding.

An inference graph also cannot establish which training-only losses, dropout, optimizer, initialization or normalization schedules produced its weights. There is no basis to say that V3 added an undocumented silence loss, peak penalty, short-window loss, or teacher distillation.

## Silence, peaks and the exported audio

The Python inference [core, lines 481-482](https://github.com/supertone-oss-archive/supertonic-py/blob/main/supertonic/core.py#L481-L482) returns the vocoder's raw floating-point output. There is no amplitude normalization or clipping on this return path. The high-level [pipeline, lines 277-298](https://github.com/supertone-oss-archive/supertonic-py/blob/main/supertonic/pipeline.py#L277-L298) inserts arrays of exact zeros between text chunks. Those externally inserted gaps do not demonstrate decoder reconstruction of encoded silence.

The browser [WAV writer, lines 522-526](https://github.com/supertone-oss-archive/supertonic/blob/main/web/helper.js#L522-L526) explicitly clamps each sample to [-1,1], then multiplies by 32767 for PCM16. This is sample clipping, not whole-utterance peak normalization, and it is outside the neural decoder. The Python [save method, line 327](https://github.com/supertone-oss-archive/supertonic-py/blob/main/supertonic/pipeline.py#L327) delegates directly to `soundfile.write`; WAV defaults to [PCM16](https://python-soundfile.readthedocs.io/en/latest/#soundfile.default_subtype). Bounded saved audio therefore cannot prove that the raw graph avoids overshoot. No claim is made that it actually overshoots on our panel, because that comparison was not run.

Both examined official repositories now redirect to the company's `supertone-oss-archive` organization and report archival on September 9, 2026. This changes source availability, not the graph identity verified above. The sources remain inspectable. [SDK repository status](https://github.com/supertone-oss-archive/supertonic-py).

## What explains the apparent advantage

**Inference from the evidence:** Supertonic can move most learned processing to a low frame rate and emit blocks directly. A representation learned for that decoder can make its rendering task easier. Our encoder is intentionally frozen: its latent space was learned together with AudioVAE2's original decoder, not this direct-block decoder. Preserving all input coordinates in an adapter does not establish equally easy decoding or matching inductive biases. This is a plausible transfer difficulty, not proof that our architecture lacks sufficient capacity.

The existing loss-balance diagnosis is an independent, measured training problem. A large quiet/active gradient imbalance cannot be excused by Supertonic's success, and repairing the objective adds no deployed layers. Conversely, the absence of tanh or a sample-rate filter in Supertonic does not prove our observed peaks and repeating residual must disappear with more training.

The useful borrowing remains the efficient body/head, compatible fixed normalization, and training-only perceptual supervision. Compare teacher and student on the same raw latents, float outputs, causal history, masks and held-out sources. Correct objective conditioning first; assess remaining errors before deciding whether a small architectural adaptation is required. Public TTS speed figures and attractive demos are not evidence of performance equivalence with Mimi or AudioVAE2 on our codec benchmark.
