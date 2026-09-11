# Supertonic methods for the AudioVAE2-conditioned student

Reviewed 9 September 2026. This is a source comparison and recommendation. No training, model execution, performance benchmarks or decoder changes were made.

The recommended combination remains the current causal Supertonic-style waveform body, the frozen AudioVAE2 encoder and its exact raw latents, and the frozen AudioVAE2 decoder as the target. Continue from the saved step-8,090 student. The evidence does not identify another missing Supertonic inference layer that handles our silence, peak or expressive failures.

## Verified release versus published recipe

The [released Supertonic 3 configuration](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/tts.json) and inspected [vocoder graph](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/vocoder.onnx) establish ten 512-channel causal ConvNeXt blocks with 2048-channel expansion and a nonlinear direct-waveform head. The graph uses LayerNorm, GELU, residual scales and PReLU. There is no Snake, iSTFT, terminal tanh, clamp, sample-rate post-filter or silence gate. Our current student already contains the corresponding main body and head, adapted to 48 kHz.

The [earlier Supertonic paper](https://arxiv.org/html/2503.23108v3#S3.SS1) describes a jointly trained speech encoder and decoder, multi-resolution mel reconstruction, multi-period and multi-resolution discriminators, and feature matching. It also describes BatchNorm at the stem and after the block stack. These mechanisms are present in our current recipe. Our two normalization sites were calibrated with fixed weights at step 500 and their statistics frozen; GAN and feature matching subsequently trained the parent. The earlier affine-only student report is historical.

The paper is not a full specification of Supertonic 3 training. The official release exposes inference code and assets, not the V3 autoencoder trainer, encoder checkpoint, discriminator checkpoint or specialized event sampling. In particular, V3 metadata has an encoder input width different from the paper's mel-only description. Do not infer the full V3 encoder implementation from the older paper.

## How this relates to our failures

| Concern | Source evidence and current status | Practical adaptation |
| --- | --- | --- |
| Persistent quiet residual | No separate suppression stage in the released decoder. Our calibrated normalization is already present. | Compare the student's response to actual encoded quiet histories with the teacher's response; inspect absolute residuals and loss gradients. |
| Peak overshoot | The released head is unbounded. It does not establish guaranteed peak control. | Tanh is an AudioVAE2-derived candidate, not a missing Supertonic component. It requires adaptation and has not yet passed quality preservation. |
| Transients and waveform texture | The paper uses mel, period/spectral discriminators and feature matching; these already run in our base. | Correct changed-loss gradient calibration before interpreting the short-window or complex-discriminator comparisons. |
| Laughter, whistles and transitions | No public category-specific V3 decoder recipe was located. | Qualify actual event intervals and ensure they reach both reconstruction and discriminator views. |
| Low CPU cost | Most temporal processing is at the internal frame rate, followed by a direct waveform head. | Preserve this body; calibration and sampling corrections add no inference work. |

One apparent silence treatment belongs to the application: the [official Python helper](https://github.com/supertone-inc/supertonic/blob/main/py/helper.py#L200) returns the vocoder waveform directly, then inserts literal zero samples between separately synthesized text chunks, with a default gap of 0.3 seconds. This cannot establish the raw decoder's silence reconstruction quality and would alter continuous codec output if copied as a suppression rule.

A newer related Supertone paper, [RobustSpeechFlow](https://arxiv.org/html/2605.22083v1#S3.SS3.SSS3), constructs a silence representation by encoding a zero-padded waveform and repeating the resulting latent frames. Its application is text-to-latent skip augmentation, not decoder noise suppression. The transferable principle is to use actual encoded silence instead of assuming a zero latent vector. For our stateful codec, encode continuous quiet and speech-to-quiet examples with the frozen AudioVAE2 encoder and use its decoder's output as the target; do not splice an arbitrary constant latent and assume a realistic transition. This supports the diagnostic direction but does not establish a new proven decoder loss.

## Why sharing architecture does not make the tasks identical

The published Supertonic autoencoder can learn its encoder representation jointly with its decoder. Our encoder is deliberately fixed. Our interface supplies 64 coordinates once every 40 ms; the student preserves them in four internal phase representations and produces 1,920 output samples. The Supertonic release uses 24-channel frames associated with 512 samples at 44.1 kHz, approximately 86 frames per second.

The student therefore needs to learn a different latent-to-waveform mapping. This is a plausible source of different learning behavior, not proof that its adapter is defective or insufficient. The original AudioVAE2 decoder demonstrates that useful reconstruction is possible from these latents. It does not prove equal expressiveness or sample efficiency for every cheaper decoder.

Retain the exact raw-latent contract. Do not copy Supertonic's latent statistics, add framewise normalization to AudioVAE2 means without an invertibility analysis, or assume zero latent values represent silence. Arbitrary teacher/student hidden layers also need not share coordinates or temporal semantics. Their direct equality is not an omitted distillation requirement.

## Additional sampling issue found in this review

The current training code selects one uniformly random 9,120-sample, 190 ms discriminator view from each eligible full reconstruction crop. Student and teacher use the same view. This follows the paper's duration and is not intrinsically incorrect.

However, a full crop labeled as laughter can contain a short laugh that the 190 ms slice misses. The same applies to peak transients, breaths and speech-to-silence transitions. The previous data audit qualified source labels and full-crop durations, not actual event occupancy in every discriminator input. Corrected experiments should record both. Include a declared mixture of ordinary random views and event-targeted views; do not silently claim that every sample from an expressive source teaches the desired event.

## Recommended sequence

1. Retain step 8,090, all trained student weights, optimizer moments and fixed model-normalization statistics. Encoder and teacher remain frozen.
2. Build verified event intervals and held-out coverage. Record event duration in full reconstruction crops and in the selected discriminator views, alongside ordinary multilingual speech.
3. Recalibrate only the gradient-balancing state of changed objectives, using fixed student weights. The previous complex branch received only 0.27% combined perceptual output-gradient norm contribution against a nominal 30%; its comparison remains inconclusive. See the [experiment follow-up audit](../convnext-fusion-experiments/follow-up-audit.md).
4. Compare teacher and student using identical real latents, valid lengths and histories. Measure quiet floor, DC, 480- and 1,920-sample phase patterns, transients and peaks. Separate student-to-teacher reconstruction from teacher-to-original quality.
5. If gradients and coverage are sound but quiet errors remain, test whether the existing waveform head can fit the teacher's quiet response while preserving speech. This identifies whether another architecture experiment is warranted; it is not a guaranteed fix.

No public evidence guarantees Supertonic 3-level RTF and AudioVAE2 quality for this student. The proposed training corrections preserve the current decoder cost; their quality benefit must be measured. The original training run remains paused.
