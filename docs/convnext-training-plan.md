# Convnext: lightweight AudioVAE2 decoder distillation

Plan approved for initial implementation, 8 September 2026. Branch: `Convnext`, created from `main` at `7150ae9bc4aff725b303c33b2417bd50d58a4f71`.

Current recipe, 9 September 2026: [the corrected recipe v2](convnext-recipe-v2.md) supersedes the architecture-adapter, normalization and stage-entry choices in the historical plan below. It uses an identity-preserving four-phase adapter, sample/frame-weighted reconstruction, fixed-weight normalization calibration after 500 updates, and scheduled adversarial and feature-matching training from update 501. The 0.99 correlation target is a final reconstruction goal, not a prerequisite for these losses. The previous mel-cap and quiet-phase comparisons are complete and their candidates were rejected. Their checkpoints and evidence remain preserved.

The new finite pilot plan contains 320,000 distinct optimizer windows (206.54 scored hours), plus 512 separate training-only calibration windows. The new recipe has passed local, Runpod CPU and full-model H100 correctness checks. Its 10,000-update pilot is running as `decoder-recipe-v2`; the full evidence and launch identity are recorded separately. This remains an experimental decoder, with no claim of qualified quality or RTF.

The sections below retain the original proposal and its historical decisions. Use the linked recipe v2 for the current training schedule and implementation.

The user clarified data reuse on 9 September: independent diagnostic experiments may reuse downloaded training audio, including audio used by a discarded earlier model. Do not repeat scored audio intervals within one run or its resumed continuation. Matched comparison arms may use the same ordered examples once in each arm. Preserve development/test exclusions, source identities, the exposure journal and necessary overlapping causal context. The earlier global exclusion of all audio seen by any experiment is superseded; do not rewrite its historical manifests or results. Before a final training run, freeze the recipe and a deduplicated manifest with its own exposure ledger.

## 1. Decision and scope

Freeze the original AudioVAE2 encoder and original decoder. Independently implement a lighter replacement decoder, using Supertonic's architectural ideas as a reference. Initialize all student weights afresh and train using AudioVAE2 as the sole teacher. The original decoder is needed during training, but not by the deployed student.

The computational change is to move most learned processing into a low-rate causal ConvNeXt body and use a direct waveform projection, replacing AudioVAE2's expensive progressive decoding path. Keep the same encoder and latent interface. The proposed ten-block, 512/2048-channel body preserves the capacity previously requested; it is a starting design for this lighter AudioVAE2 decoder, not a requirement to reproduce Supertonic numerically.

The target is teacher-level audible quality with CPU decoding close to Supertonic. Neither quality parity nor a particular RTF is established before training and measurement.

Fixed decisions:

- Preserve the AudioVAE2 latent contract: raw posterior mean, 64 channels, 25 frames per second, existing ordering, gain and frame alignment.
- Keep all ten Supertonic blocks, 512 hidden channels and 2,048-channel expansions. No pruning, grouped replacements, reduced widths or fewer blocks.
- Preserve the nonlinear head's 2,048-channel hidden layer and shared PReLU activation.
- Supertonic contributes architectural reference only. Do not import its weights, normalization statistics, code or graph; do not train on its outputs or activations. There is no Supertonic clone, weight-conversion task or faithful-reconstruction gate.
- Keep the existing 16 kHz mono input and 48 kHz mono output interface. This is the previously accepted decoder-replacement exception; it is not a native 48-to-48 codec.
- Target the original decoder's default `sr_cond=48000` behavior explicitly. This parameter conditions bandwidth; it does not change the physical 48 kHz output sample rate. Other conditioning bins must retain an explicit original-decoder fallback until separately supported and validated.
- Streaming is a first-phase requirement. CPU evaluation uses one thread on Apple, Intel and AMD. GPUs are used for training and teacher preparation only.
- No new encoder, DAC teacher, video model or TTS language model is required for this experiment. The teacher is AudioVAE2. The frozen posterior does not need a new KL loss.
- The existing released runtime remains available. The student becomes selectable only after its own quality, streaming and performance validation.

Confirmed planning inputs: one H100 80 GB for the pilot, scaling only if needed, and a checkpoint intended for commercial use. GigaSpeech is excluded from the main training manifest under its current audio-access terms. The user authorized training on the existing Runpod, which is verified as an H100 NVL with 95,830 MiB. No new Pod was provisioned.

The user's existing Runpod SSH endpoint has now been verified: AMD EPYC 9654 CPU and one H100 NVL reporting 95,830 MiB. After setup it has approximately 39 GB free in the existing 100 GB workspace. The training environment uses PyTorch 2.11.0+cu128 with native Muon and TensorBoard 2.21.0; the original CPU validation environment uses PyTorch 2.8.0+cu128. Use this existing resource for preflight; no new Pod has been provisioned. Full FP32 teacher-waveform caching for 100 hours exceeds its current free space, so sustained storage planning must use bounded shards/online teacher decoding or an explicitly arranged expansion.

## 2. Evidence and what remains unknown

Use the published Supertonic architecture and previous graph inspection to understand efficient design choices. Implement our own student directly for AudioVAE2's interface. The older Supertonic paper supplies architecture and training context, but is not a complete V3 training release. It reports 11,167 training hours, 1.5 million updates and four RTX 4090 GPUs; these numbers do not establish the data or H100 time needed for this decoder-only distillation run. [Paper](https://arxiv.org/html/2503.23108v3#S4).

The previously inspected graph is `work/supertonic3_assets/vocoder.onnx` in the surrounding research workspace. Its recorded SHA-256 is `085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba`. This identifies the architectural evidence, not an asset to convert or include in the student. [Released graph](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/vocoder.onnx), [configuration](https://huggingface.co/Supertone/supertonic-3/blob/main/onnx/tts.json).

The current project's AudioVAE2 provenance points to VoxCPM2 revision `32279effe8c19989596f05d353d1447f51d9e915` and source revision `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`. The original cached `audiovae.pth` was hash-verified during this planning audit: SHA-256 `94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1`. The matching `audio_vae_v2.py` hash is `2efdff1708d8ec1471624aae6f232d0f933de26b788b6901c434246847e2d3a8`. The existing strict loader uses that source's `AudioVAEConfig()` defaults; serialize those defaults beside the hashes before target generation. Do not accidentally load the nearby `work/causal_assets/audiovae.pth`: that is VoxCPM 1.5. [Model revision](https://huggingface.co/openbmb/VoxCPM2/tree/32279effe8c19989596f05d353d1447f51d9e915), [AudioVAE2 source](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py).

An unofficial [Supertonic training repository](https://github.com/ORI-Muchim/supertonictts-training) provides conversion and GAN-training examples for earlier versions. It is background research only, not a source of code, weights or training defaults for this implementation.

The previous shared ONNX 1.29 experiment measured full-file decoder RTF 0.01383 for Supertonic and 0.09542 for Pocket continuous Mimi on one Apple thread. Supertonic used synthetic latents. That is historical throughput evidence, not a measured streaming target or a quality comparison. New stateful measurements must use matched durations, thread settings and explicit decoder scope.

## 3. Proposed student architecture

```text
16 kHz waveform
    -> frozen AudioVAE2 encoder
    -> mu [B, 64, T] at 25 Hz
    -> pointwise phase adapter: Conv1d(64, 256, kernel=1)
    -> temporal pixel shuffle [B, 64, 4T] at 100 Hz
    -> causal embedding Conv1d(64, 512, kernel=7)
    -> 10 independently implemented causal ConvNeXt blocks, 512 / 2048
    -> trainable channelwise affine, initialized to identity
    -> causal Conv1d(512, 2048, kernel=3)
    -> shared-slope PReLU
    -> bias-free Conv1d(2048, 480, kernel=1)
    -> waveform reshape [B, 1, 1920T] at 48 kHz
```

The phase adapter is the proposed new component. For each available latent it predicts four ordered internal frames without accessing the next latent. It does not compress the 64 channels to 24, repeat an identical frame, interpolate through future frames, or change external latency into 10 ms.

| Component | Observed architectural reference | Proposed independent implementation |
|---|---|---|
| Input stem | Biased 24-to-512 convolution, kernel 7 | New biased 64-to-512 convolution, same kernel |
| Temporal blocks | Ten depthwise convolutions, kernel 7, groups 512 | Preserve |
| Dilations | `[1,2,4,1,2,4,1,1,1,1]` | Preserve |
| Channel mixing | 512-to-2048-to-512 in every block | Preserve |
| LayerNorm | Channels within each frame, epsilon `1e-6` | Preserve |
| GELU | Exact Erf formulation | Preserve; no tanh approximation initially |
| Residual path | Learned per-channel layer scale and addition | Preserve |
| Final normalization | Affine inference function from fixed running statistics | New channelwise affine, scale 1 and bias 0; no imported statistics |
| Head hidden convolution | Biased 512-to-2048, kernel 3 | Preserve |
| PReLU | One shared learned slope | Preserve; do not substitute 2,048 slopes |
| Output projection | Bias-free 2048-to-512 | New bias-free 2048-to-480 |
| Temporal padding | Left edge replication | Preserve at real stream starts; cache history afterward |
| Final waveform | Reshape, no terminal clamp or tanh | Preserve |

Use a new biased input convolution without a separate stem BatchNorm. The architectural reference already has a biased affine stem at inference. Supertonic's TTS-specific six-frame packing, division by 0.25 and 24-channel mean/std conversion do not apply to AudioVAE2 latents.

There is no Snake, GRN, iSTFT or iterative refinement in this baseline. `convnext_2` in the TTS configuration is not a specification of ConvNeXt V2.

Derived convolution arithmetic is approximately 2.537 GMAC per audio-second versus 2.178 for the original Supertonic decoder, about 16.5% higher. This comes mainly from the 100 Hz clock instead of 86.13 Hz. These counts exclude memory traffic, normalization and nonlinearities and are not RTF predictions.

### Initialization

Implement the student directly at 64 input channels and 48 kHz output. Initialize all convolutional and linear weights with a declared seeded initialization; initialize LayerNorm scale/bias to 1/0, residual layer scale to a proposed `1e-6`, and the shared PReLU slope to a proposed `0.25`. No parameter, activation target or normalization statistic comes from Supertonic.

All student parameters are trainable. Start with one proposed AdamW learning-rate schedule toward `2e-4`, with warm-up, rather than separate transferred/new parameter groups. The tiny-set overfit preflight must establish gradient flow and trainability before the pilot. Fresh initialization is the main experiment, not an optional control.

Use a learned per-channel affine before the waveform head, initially scale 1 and bias 0. It has the same function class as fixed-statistics normalization at inference, without inherited running statistics or future-dependent batch/time normalization during training. It can be folded into a following convolution at export after validating equivalence. Do not introduce BatchNorm or SyncBatchNorm statistics.

The earlier claim that this student inherits Supertonic's model terms was tied to an unintended weight-transfer plan. That plan is removed. Architectural inspiration alone is not the weight/output transfer described by the cited model license. Our implementation and freshly trained student use AudioVAE2 teacher supervision only. VoxCPM2 is released under Apache-2.0; preserve applicable teacher, dependency and dataset notices rather than assigning a Supertonic-derived license to this checkpoint. [Supertonic license definition](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE), [VoxCPM2 license statement](https://huggingface.co/openbmb/VoxCPM2#license).

## 4. Teacher and target construction

For a source recording `x`:

1. Preserve the original waveform, actual sample rate and bandwidth provenance.
2. Apply the pinned preprocessing to obtain the 16 kHz encoder input.
3. Freeze encoder weights and buffers; call the exact posterior-mean encoding path.
4. Generate `y_teacher = original_decoder(mu, sr_cond=48000)` from the same latents.
5. Feed those latents to the student and compare aligned outputs.

Use the original FP32 teacher, not the optimized INT8 decoder. Target generation uses evaluation mode, disabled parameter gradients, no autocast and an explicit TF32 policy. Verify the GPU target path against the pinned reference before preparing the corpus. Slight cross-device floating-point differences need declared tolerances; do not call them bitwise identity.

Compute latents and teacher outputs with continuous utterance context or verified stateful execution. Do not independently encode arbitrary short crops from a reset state. Cache identity includes source checksum, encoder/decoder hashes, configuration, preprocessing, posterior policy, conditioning and sample offsets.

Do not cache all hidden teacher layers. Begin with latents and waveform targets; selected features can be added later if they provide a measurable benefit. Transcripts are not required for reconstruction training.

This first qualification covers encoder-produced latents. Before advertising the student as a drop-in decoder for the complete VoxCPM TTS system, evaluate a separate panel of latents produced by its speech generator. Those latents can differ from real-audio posterior means. If that panel regresses, add a separately identified teacher-supervised generated-latent allocation after the codec pilot; do not silently claim TTS compatibility from reconstruction scores alone.

### Bandwidth-aware supervision

LibriSpeech, GigaSpeech, FLEURS and MLS normally provide 16 kHz audio. Upsampling those recordings to 48 kHz does not create authentic content above 8 kHz. Treating those upsampled signals as fullband real targets would penalize the high-frequency synthesis we want to preserve.

| Source evidence | Original-recording loss | Teacher loss | Adversarial reference |
|---|---|---|---|
| 16 kHz source | Compare only the supported speech band, including a 16 kHz reconstruction view | Full 48 kHz teacher output | Band-match real and generated signals; no fake fullband ground truth |
| Native 44.1/48 kHz source | Use valid native bandwidth after documented alignment/resampling | Full teacher target as an additional anchor | Band-matched native reference; 48 kHz material supplies the widest reference band |
| Restored/enhanced source | Mark provenance and cap exposure | Teacher anchor | Separate source accounting; not called pristine recorded highband |

Even with fullband originals, the frozen encoder sees only 16 kHz audio. We seek perceptually faithful highband reconstruction, not guaranteed recovery of original phase-specific information above 8 kHz. Avoid making highband waveform MSE the dominant objective.

Apply any gain/noise augmentation before both encoder and teacher, or cache matching variants. Never change student inputs while silently retaining incompatible teacher targets. Do not apply utterance-level gain fitting, silence removal or fitted time shifts during evaluation.

## 5. Data acquisition and language balance

The earlier 500-hour acquisition and 10,000-update continuation have finished and are historical experiments. Their allocation included FLEURS across all 102 configurations, LibriSpeech, original IndicVoices across all 22 scheduled Indian languages and expressive recordings. Do not treat that earlier checkpoint, batch size or validation schedule as the current recipe. The corrected 1,009-update pilot used a fresh student and batch size 32, with scheduled fixed-panel evaluation. For the next diagnostics, restore eligible training audio from the downloaded corpus under the run-specific reuse policy above, then publish achieved language/event coverage after development, test and parent-run exclusions. Whole-utterance teacher targets retain their bounded rolling cache.

The table below is the earlier 100/1,000-hour proposal, retained for reference. It does not describe the active acquisition or authorize a further expansion.

| Source | Pilot hours | Main hours | Purpose |
|---|---:|---:|---|
| LibriSpeech train | 10 | 60 | Clean and difficult English read speech, 16 kHz |
| Common Voice Scripted 26.0 English train | 5 | 50 | Diverse accents, including Indian English, and recording conditions |
| VoxPopuli English transcribed train | 5 | 50 | Parliamentary speech and more speakers, 16 kHz |
| FLEURS train only | 20 | 200 | Breadth across 102 language configurations, 16 kHz |
| IndicVoices original, verified 44.1 kHz subset | 20 | 250 | Hindi and other Indian languages; natural voices and fullband references |
| IndicVoices-R, disjoint originals | 5 | 50 | Cleaner Indic speech; enhanced 48 kHz references |
| Hi-Fi TTS train | 20 | 150 | Clean 44.1 kHz anchor |
| VCTK 0.92 training speakers | 5 | 20 | Genuine 48 kHz speech, varied speakers and accents |
| MLS non-English train | 5 | 120 | More depth in European languages, 16 kHz |
| AISHELL-3 train | 5 | 50 | Mandarin, 44.1 kHz |
| **Total** | **100** | **1,000** | Unique hours after filtering and deduplication |

The 1,000-hour proposal contains 470 hours selected from original 44.1/48 kHz recordings, 50 enhanced hours and 480 hours allocated to speech-band/diversity supervision. Common Voice sample rates and bandwidth vary and are not counted as native highband references by default. Actual usable bandwidth must be checked; a high sample-rate header alone is not evidence of highband content.

FLEURS has 102 configurations; IndicVoices covers 22 scheduled Indian languages, with overlap. Compute the actual language union from the manifest. Do not describe the result as literally all languages. Reserve at least 50 main-pool hours for Hindi within the Indic allocation. The FLEURS pilot provides only about 12 minutes per language if balanced and cannot establish multilingual convergence by itself.

Separate sampling exposure from unique hours. Proposed exposure targets are roughly 30% English, 30% Indic and 40% other languages, jointly constrained to at least 40% original high-rate-reference batches. Check feasibility against the manifest, cap repeated low-resource examples and limit domination by individual narrators. Sample speakers/session groups within languages. Track achieved rather than just requested weights.

Preserve quiet speech, breathy voices, fricatives, plosives, expressive pitch, accents, natural silences and moderate background sound. Allocate 5-10% of exposure to identified difficult conditions rather than filtering everything to studio speech. Avoid synthetic enhancement as a default cleanup step.

### Acquisition facts and conditions

- [LibriSpeech](https://www.openslr.org/12/): 16 kHz, CC-BY-4.0. Use official train partitions.
- [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech): excluded because its audio Terms of Access restrict use to noncommercial research and education. The Apache metadata badge does not supersede those terms. The user has selected commercial usability, so do not acquire it for this main run or mix its targets into the replacement source slot.
- [Common Voice Scripted 26.0](https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech/cv-corpus-26.0-2026-06-12.json): CC0 audio, MP3/TSV, sufficient English train material for the selected quota. Inspect per-file bandwidth. Preserve speaker/client groups and accent metadata. [Official catalog](https://commonvoice.mozilla.org/os/datasets).
- [VoxPopuli](https://github.com/facebookresearch/voxpopuli): its license table distinguishes CC0 data from noncommercial code/models. Use the English audio through our own generic loader, not its model or training code. The data provides ample English material at 16 kHz. Keep source acknowledgments and item-specific provenance. Neither replacement exactly reproduces GigaSpeech's podcast/YouTube distribution.
- [FLEURS](https://huggingface.co/datasets/google/fleurs): 16 kHz, CC-BY-4.0. Use train only, never the existing benchmark clips or official dev/test recordings.
- [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices): CC-BY-4.0 with a contact-sharing access gate. Originals include both 44.1 kHz and 8 kHz, so explicitly select verified 44.1 kHz sources. [Rate evidence](https://arxiv.org/html/2409.05356v1#S3.SS2).
- [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r): CC-BY-4.0, 48 kHz, restored with separation/enhancement models. Flag it as enhanced and keep original/restored versions in the same split.
- [Hi-Fi TTS](https://www.openslr.org/109/): 44.1 kHz, CC-BY-4.0, only ten speakers. Useful fidelity anchor, not a substitute for speaker diversity.
- [VCTK 0.92](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full): 48 kHz, CC-BY-4.0. Select one microphone copy and split by speaker.
- [MLS](https://www.openslr.org/94/): CC-BY-4.0, eight languages, 16 kHz. Use non-English train portions.
- [AISHELL-3](https://www.openslr.org/93/): Apache-2.0, Mandarin. [Official format description](https://aishell-3.oss-cn-beijing.aliyuncs.com/AISHELL-3%20ReadMe.pdf) specifies 44.1 kHz.

Common Voice Spontaneous 4.0 can supply a later small substitution, but its current English release has only about 7.81 validated hours across splits, so it is not a credible 100-hour replacement by itself. [Exact release card](https://mozilladatacollective.com/datasets/cmqialpeo0077nr077xqdqo0j). LibriTTS-R can replace some clean English material, but is a restored 24 kHz derivative of the same LibriSpeech family, not independent fullband evidence. Access-gated data is acquired only after the user's access is established; do not submit contact details or accept new access conditions silently.

### Split and manifest requirements

Split by speaker and original recording/session before cropping. Use recording/podcast identity when speaker IDs are missing. Cross-check related LibriVox sources across LibriSpeech, Hi-Fi TTS and MLS, and original/restored Indic recordings. Use identifiers and audio fingerprints to prevent duplicate train/test material, including alternate encodings and microphone copies.

Manifest fields: dataset revision, source ID and checksum, parent recording, speaker/session, language, original rate, measured bandwidth class, enhancement status, duration, split, attribution/access record, gain/resampler policy and cache key. Sensitive access credentials stay out of manifests and Git.

The current 60-clip multilingual comparison is a regression panel already used in research decisions. Preserve it by hash and parent recording, but add an untouched final test. Proposed development set: 20 clips per FLEURS language plus disjoint fullband speakers; proposed sealed test: another disjoint 20 clips per language plus at least 100 native fullband/difficult clips where source availability permits. Use official train/dev versus test speaker separation and parent-source exclusion. Do not claim to know or remove all of the teacher's pretraining overlap.

## 6. Losses and training sequence

### Optimizer decision

The implementation preflight supports AdamW and the Muon/AdamW hybrid. Before selecting the sustained pilot optimizer, compare all-AdamW with simultaneous Muon on the twenty 2D hidden channel-mixing matrices and AdamW on the remaining parameters. AudioVAE2 remains the only teacher in both arms. Keep discriminators on AdamW initially to isolate the generator optimizer change. Native Muon parameter partitioning, CPU checkpoint replay and dashboard logging passed in the compatible Runpod training environment. Each optimizer also completed 32 full-capacity synthetic H100 updates. This establishes implementation wiring, not a measured speech convergence advantage. Select by held-out quality per elapsed H100 hour, including optimizer overhead, and preserve both optimizer states. See [the optimizer research and comparison design](convnext-optimizer.md).

### Reference recipe

The earlier Supertonic recipe uses multiresolution mel L1, least-squares GAN and discriminator feature matching with generator coefficients 45, 1 and 0.1. Generator mel FFTs are 1024/2048/4096, with 64/128/128 mel bands. MPD periods are 2/3/5/7/11; MRD FFTs are 512/1024/2048. These are literature-based starting choices for our independently implemented losses. Specify their equations, reductions and discriminator real/fake targets explicitly for reproducibility. Initialize and train our own discriminators. [Training equations and appendix](https://arxiv.org/html/2503.23108v3#B1.SS1).

Keep the same sample-domain FFT sizes as the first 48 kHz baseline and document the resulting change in time support from 44.1 kHz. Any time-preserving rescaling is a separate loss configuration. Adversarial crops of about 0.19 seconds are taken from the valid generated region; that is not the entire generator context length.

### Proposed objective

```text
L_generator = 45 * L_original_mel_valid_band
            + lambda_teacher * L_teacher_multiresolution_spectral
            + lambda_wave * L_teacher_aligned_waveform_L1
            + 1 * L_adversarial
            + 0.1 * L_discriminator_features
            + lambda_KD * L_selected_teacher_features
            + lambda_SSR * L_reconstructed_speech_features
```

The teacher and waveform coefficients are new hyperparameters. A concrete initial configuration is `lambda_teacher=15`, `lambda_wave=1`, and both optional feature coefficients zero. These numbers are proposals, not published guarantees. Define normalization over valid bands, samples and resolutions before interpreting weights. Inspect gradient contributions in warm-up and adjust at most the combined teacher-anchor strength before fixing a pilot configuration; do not start a large sweep.

Warm-up omits GAN and optional feature losses. Later perceptual training adds the reference discriminators. Waveform L1 compares aligned teacher/student outputs from identical latents and supplements spectral losses; it must not overwhelm real-recording or perceptual supervision.

### Quality improvements, in order

1. Establish the full-capacity spectral/waveform-distilled GAN baseline.
2. If quality remains below the teacher, test one or two selected teacher-feature targets with training-only channel/time adapters. AudioVAE2 and Supertonic layers do not correspond one-to-one. Do not arbitrarily regress every hidden tensor. [DLL-APNet](https://arxiv.org/html/2509.13667v1#S3.SS3).
3. Test a multilingual speech-representation reconstruction loss if intelligibility or phonetic detail is the residual weakness. The reference branch has no gradients; the student-audio branch must retain gradients through the frozen feature model. This adds training memory and compute, but no inference model. [JHCodec](https://arxiv.org/html/2603.05887v1#S2.SS4).
4. Use transient/excess-energy penalties only if residual listening errors justify them. Check that noise suppression does not erase quiet consonants. [LDCodec](https://arxiv.org/html/2510.15364v1#S2.SS4).

These are candidate improvements, not measured gains in this project. Keep data exposure, architecture and evaluation fixed when comparing them. Do not introduce Snake/ADAA, diffusion refinement or a second inference network by default.

## 7. Streaming and crop correctness

For this adapter and dilation schedule, required student past context is `6 + 6*sum(dilations) + 2 = 116` internal frames, equivalent to 29 input latent frames or 1.16 seconds. It is history, not lookahead. Estimated convolution state is 56,704 FP32 scalars, about 222 KiB per stream, excluding workspaces.

A proposed training example contains 64 scored latent frames, or 2.56 seconds, preceded by 29 context frames when available. Losses exclude the context-only region and padded tail. Cache the encoder's continuous context independently; the student's 29-frame requirement does not justify resetting the teacher encoder there.

Teacher output windows come from continuous teacher execution. The student receives enough matching history to reproduce interior behavior. Only real utterance starts use initial edge replication. Include short utterances and genuine resets deliberately rather than turning every random crop into a fake stream start.

Acceptance checks cover 1/2/4 latent-frame calls, irregular partitions, empty input, resets, interleaved streams, long streams and final partial input. Every complete input frame consumes 640 input samples and represents 1,920 output samples. For native 16 kHz input length N, the complete codec uses original-length metadata to remove only the documented final padding and return 3N output samples. Raw decoding from T latent frames without that metadata returns 1920T samples; it cannot infer the original partial-frame length. Verify both contracts against the current wrapper rather than inventing a fitted trim.

Check exact shape/sample accounting; numerical full-versus-streaming equivalence; unchanged prefixes when future latents are modified; correct subframe ordering; fixed normalization; and absence of chunk-boundary gain or phase changes. The input API remains 40 ms latent granularity. Mimi comparisons use its supported 80/160 ms modes.

## 8. Work phases and decision gates

| Phase | Deliverable after plan review | Required evidence before continuing |
|---|---|---|
| A. AudioVAE2 contract and cost budget | Pinned encoder/teacher interface, sample-alignment fixtures and proposed student operation budget | Correct 64-channel latents, frame rate, conditioning and tail contract; fresh initialization specified; no dependency on Supertonic assets |
| B. Full student and streaming | Adapted model, stream state and CPU export | Causality, shapes, sample counts and whole/stream agreement; early one-thread costs on Apple/Intel/AMD |
| C. Data and teacher cache | Versioned 100-hour manifest and targets | Access conditions resolved; no test leakage; correct source bandwidth; cache/reference agreement |
| D. Training preflight | Tiny-set overfit, checkpoint replay and complete-step timing | Finite gradients, falling training reconstruction loss and successful tiny-set overfit; working save/resume; measured memory and cost forecast |
| E. 100-hour pilot | Full student learning curve and listening outputs | Quality improving across languages and conditions; no stream defects; compute within the chosen budget |
| F. Main pool | Initially 1,000 hours, bounded continuation | Pilot justifies data expansion; fixed recipe and measured Runpod allocation |
| G. Targeted quality addition | At most one new loss hypothesis at a time | Improvement survives held-out tests without extra inference computation |
| H. Candidate delivery | Opt-in exported model and quality/RTF report | Quality, streaming and performance gates all pass; no silent replacement of the stable runtime |

Phase B timing is a structural-cost screen, not a trained quality result. Use actual AudioVAE2 latent shapes and distinguish untrained outputs. Record warm-up, CPU identity/ISA, runtime, state initialization, p50/p95/p99, output duration and failures. Do not infer H100 training speed from these CPU timings.

Initial proposed exposure budgets: 5,000 reconstruction-only updates plus 20,000 perceptual updates for the feasibility pilot, with global batch 32 and 2.56 scored seconds/example if memory allows. This represents about 569 audio-hours of scored exposure, not 25,000 unique clips or proven convergence. These are from-scratch student budgets; no pretrained Supertonic convergence advantage is assumed, and 25,000 updates may be insufficient for quality parity. A possible main continuation is 150,000-300,000 updates at global batch 64, about 6,827-13,653 hours of exposure. Re-estimate after learning curves; do not run to a large arbitrary step count when quality plateaus.

## 9. Runpod plan

The user selected one H100 for the pilot. The existing endpoint is now verified as an H100 NVL with 95,830 MiB, and bounded training checks have run there. Continue on that existing Pod; no new Pod was provisioned. The bootstrap corpus, teacher target parity and first 1,000 speech updates are complete. The scored training updates totaled 166 seconds, excluding validation and checkpoint overhead; this is not an end-to-end pilot time estimate. Broader data and perceptual training remain pending.

Use a persistent network volume attached when the Pod is created. It survives Pod deletion; an ordinary Pod volume does not. Keep a separate backup of manifests and valuable checkpoints. [Runpod storage](https://docs.runpod.io/pods/storage/types).

```text
/workspace/convnext/
  source/<commit>/
  manifests/
  originals/
  latents/<teacher-hash>/
  teacher-audio/<teacher-hash>/
  runs/<run-id>/
  exports/<run-id>/
```

Use bounded sequential shards with local hot-cache/prefetch where available. Prepare and validate data before renting GPU time for a sustained run. Avoid large numbers of tiny network-storage files.

| Uncompressed payload | 100 hours | 1,000 hours |
|---|---:|---:|
| FP32 64-channel, 25 Hz latents | 2.304 GB | 23.04 GB |
| FP32 mono 48 kHz teacher waveform | 69.12 GB | 691.2 GB |
| Both, excluding originals/checkpoints | 71.424 GB | 714.24 GB |

A 250-500 GB persistent pilot volume is a planning envelope, not an allocation request; confirm actual selected archive sizes. Main-pool provisioning depends on cached versus online teacher decoding and original compression. Do not automatically reserve multi-terabyte waveform/feature caches.

Compare two teacher strategies in preflight: cached FP32 latents plus cached waveform targets, or cached latents plus online frozen teacher decoding. Measure full-step compute and input wait. Do not reduce cache precision or use lossy teacher audio without separate target-quality evidence.

Use BF16 student autocast on H100 with FP32 parameters/optimizer state and sensitive losses as needed. Teacher target preparation remains the validated FP32 path. Keep a separate locked training environment; the runtime package's ONNX Runtime 1.29.0 / ONNX 1.22.0 baseline does not require a training-framework upgrade.

Eight H100s are a later option only if needed. That configuration would use one process per GPU and ordinary DDP on one well-connected host. Start with the same global batch/exposure and adjust gradient accumulation, rather than silently multiplying examples and learning rate. Check world-size-independent sampling and normalization. Scale only if measured utilization and communication make it useful; do not assume 8x speedup. [PyTorch distributed launcher](https://docs.pytorch.org/docs/stable/elastic/run.html).

Every resumable checkpoint contains student/discriminator/optimizer/scheduler states, optional EMA, normalization buffers, RNG and sampler state, update and audio-exposure counters, configuration, code revision, dependency/container identity, teacher hashes and data-manifest hash. Write atomically, retain latest and best, and test a resume by replaying the next batch/update. Checkpoint periodically by elapsed time as well as update count.

Report costs using measured complete updates:

```text
training_hours = updates * complete_seconds_per_update / 3600
total_wall_hours = preparation + training + validation + export/evaluation
cost = GPU_wall_hours * quoted_hourly_rate + storage + transfer
```

The complete update includes teacher strategy, discriminator updates, gradient accumulation, feature losses and I/O. No credible fixed H100-hours estimate exists until that preflight and an initial learning curve are available.

## 10. Quality and performance acceptance

All thresholds below are proposed review criteria, not current results.

**Functional gate:** zero failed sample-count, state reset, tail, prefix-causality or streaming-equivalence cases. Compare our student's full, streaming and exported implementations with each other; validate the unchanged encoder and teacher paths against their pinned AudioVAE2 references. There is no Supertonic clone-equivalence gate. The trained student is not expected to be numerically identical to its teacher.

**Objective screening:** use the current matched multilingual scorer for PESQ-WB, STOI, ESTOI, UTMOS22, DNSMOS SIG/BAK/OVRL/P.808, spectral convergence, log-magnitude error, SI-SDR and SNR. Keep native-rate audio for listening and fullband spectral measurements. Proposed teacher-relative screening margins are mean PESQ >= -0.05, STOI/ESTOI >= -0.005 and UTMOS/DNSMOS overall >= -0.05. Inspect language and difficult-condition results with uncertainty; an acceptable global mean must not hide student-specific failed clips. Reliability of learned MOS predictors across all 102 languages is not established by their availability; these metrics support listening, rather than defining audible parity. Use a fixed small development panel for frequent checks, the full development set at phase boundaries, and the sealed test only for final candidates.

**Listening gate:** blind, gain-consistent comparisons with original, teacher, student and Mimi, including quiet speech and chunk seams. Initial user review uses a compact balanced set. Before claiming public perceptual noninferiority, use a larger listener/clip study with a predeclared proposed margin of 3 points on a 100-point scale and confidence intervals accounting for listeners and clips. Do not label eight informal ratings a formal MUSHRA result. No student-only catastrophic artifacts are acceptable.

**Speed gate:** one-thread CPU streaming at 80/160 ms on Apple/Intel/AMD; report 40 ms separately where supported. Measure against both the current fast AudioVAE2 and matched Mimi decoder, targeting a substantial improvement over the former and beating the latter on each validated CPU. Supertonic-like CPU cost remains the architecture goal. A proposed within-1.5x Supertonic target can only be assessed if a comparable independently validated streaming measurement exists; implementing a stateful Supertonic clone is not a prerequisite of this project. Keep historical full-file Supertonic numbers labeled as context, not streaming evidence. If comparable streaming results become available, disclose actual chunk lengths, output sample rates and emitted-duration denominators. Record decoder-only RTF, encoder RTF, combined RTF, first-chunk compute and p95/p99 separately. Count all emitted samples and flush costs. No GPU contributes to these timings.

If speed misses, inspect scheduling/layout/runtime implementation first. If quality misses, inspect alignment, data balance, bandwidth targets and the training objective first. Reducing the full architecture or adding an expensive inference network requires a new design decision, not an automatic fallback.

## 11. Repository implementation and review boundaries

Initial implementation organization:

```text
experiments/convnext/
  audiovae_student/
    model.py       # independently implemented student and stream state
    teacher.py     # pinned frozen original AudioVAE2
    data.py        # manifest and split validation
    losses.py      # differentiable reconstruction warmup losses
    training.py    # fixed-batch preflight and checkpoint replay
    export.py      # full/stateful ONNX graphs and CPU runner
    validation.py  # full-capacity structural qualification
  tests/
  configs/         # pilot and data mixture
  requirements.txt
  requirements.lock
  README.md
```

Keep the normal installation lightweight. Training dependencies stay outside the runtime dependency set. Add an opt-in runtime integration only for a validated exported student. Keep `.env`, credentials, source audio, teacher targets and training checkpoints out of Git. Public manifests must not embed signed download URLs or secret values.

Logical implementation commits should separate: AudioVAE2 contract and student architecture/state; data/targets; training/checkpointing; export/CPU integration; validation and documentation. No bulk commit of unrelated existing files. No commit or push is performed by this planning step.

## 12. What review should settle

1. Accept the full-capacity architecture, default 48 kHz conditioning scope and unchanged 16 kHz encoder interface.
2. Use the verified existing H100 Pod for the authorized pilot; review any expansion beyond it separately.
3. Confirm acquisition readiness for the selected CC0 Common Voice/VoxPopuli replacement and the other data sources.
4. Approve the 100-hour pilot and conditional 1,000-hour expansion as budgets, rather than guaranteed convergence requirements.
5. Agree on the proposed quality/listening and streaming-speed acceptance margins before selecting checkpoints.

The frozen-teacher cache and resumable corpus trainer are implemented, and the first speech warmup is complete. Next come broader licensed data, development listening, the matched optimizer comparison and adversarial training. The current warmup is not a trained release or a quality-parity result.
