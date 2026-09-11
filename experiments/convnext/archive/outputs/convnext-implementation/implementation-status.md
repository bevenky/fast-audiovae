# Decoder implementation status

8 September 2026. Branch: `Convnext`. Source remains uncommitted.

The first real speech reconstruction warmup has completed: `warmup-speech-muon-v1`, 1,000 updates on the existing H100. The expanded corpus is being acquired directly on Runpod before continuation to step 10,000. Only the fresh 25,431,809-parameter causal student trains. It accepts the unchanged raw 64-channel AudioVAE2 posterior means and emits 48 kHz audio.

The original AudioVAE2 encoder and decoder are frozen. Their checkpoint is hash-verified, gradients are disabled, and teacher state hashes match before and after the H100 parity check. All 255 utterance targets are generated once and cached. The student optimizer contains no encoder or teacher-decoder parameters.

| Check | Result |
|---|---|
| Training speech | 230 clips, 45.85 minutes, 12 languages |
| Held-out development speech | 25 LibriSpeech clips from separate readers |
| Completed updates | 1,000, using Muon and AdamW on disjoint student parameters |
| Held-out reconstruction loss | 171.89 initially, 40.89 at step 1,000 |
| Final checkpoint | Saved at step 1,000; strict reload and finite student weights verified |
| Initial local CPU tests | 111 passed, 6 skipped, 51 subtests |
| Initial Runpod CPU tests | 116 passed, 1 skipped, 51 subtests |
| Initial whole/stream/export comparisons | 20 passed each on Apple and AMD |

CPU tests use one thread and ONNX Runtime 1.29.0. Local skips cover native Muon unavailable in PyTorch 2.8, optional teacher assets and explicitly gated CUDA. Runpod tested native Muon and actual teacher assets using PyTorch 2.11.0+cu128; its sole CPU-suite skip is the separately passed CUDA test. H100 teacher outputs also pass numerical parity against CPU on three held-out utterances.

This small LibriSpeech/FLEURS bootstrap uses public CC-BY data and excludes the reserved evaluation clips. English has two source labels in the sampler, so the current 13-label sampling is not exactly uniform over the 12 actual languages. FLEURS does not provide speaker identities. Current held-out loss uses fixed start crops from LibriSpeech, not a multilingual perceptual benchmark. All input recordings are native 16 kHz; teacher outputs supply the 48 kHz reconstruction targets.

Caching removes repeated teacher forward passes from updates. The fixed latent mapping simplifies the task, but faster convergence, final quality parity and CPU RTF remain unproven. No trained-quality or speed result against Mimi is claimed. The 166 seconds of scored training updates excludes validation and checkpoint overhead and is not the end-to-end job time.

The next phase retains the saved model, both optimizer states and the learning rate. It uses true batches of 64 and a fixed corpus of at least 500 fresh training hours. Scored segments are consumed once; overlapping causal context and padding are excluded from both loss and unique exposure. The original encoder and decoder teacher remain frozen in FP32. A bounded whole-utterance target cache avoids retaining hundreds of gigabytes of decoded teacher audio.

The acquisition plan includes 440 hours of FLEURS/LibriSpeech, 44 hours of original IndicVoices across all 22 scheduled Indian languages, and additional expressive sources. Two expressive sources have completed: 10.8649 training hours from Thorsten/JNV/JVNV/CREMA-D and 15.0498 from EmoGator. Explicitly labelled vocal events and human whistles are being added separately. Source quotas are not counted as downloaded hours. All large audio files and teacher caches stay on Runpod; the Mac holds code, metadata and a backup checkpoint.

The larger batch and aggregated finite checks measured 283.9 examples/second versus 57.7 for serial accumulation on the existing H100, with peak allocated memory of 9.38 GiB. This is a short training-step measurement with prepared targets, not complete pipeline throughput. Real training logs separate teacher/data preparation from student execution. [Batch measurement](h100-batch-profile.json).

Reconstruction validation and perceptual assessment are deferred until the longer run completes. No further optimizer comparison or adversarial phase has started. Full corpus readiness, trained quality/RTF qualification and Intel validation remain pending. The saved Intel SSH credentials still fail for this experiment.

[Open the dashboard](dashboard.md). [Frozen-teacher parity evidence](h100-teacher-parity.json). [Cache provenance](teacher-cache-preparation.json). [Completed checkpoint audit](speech-warmup-verification.json).

The expanded workflow is now running persistently on Runpod. It waits for all configured sources, verifies at least 500 actual fresh training hours and the fixed manifest, then continues the saved student to step 10,000. Its observed state is still waiting for source downloads, so new training has not started yet. Acquisition now uses four concurrent core archives and six Indic language workers. All 71 completed core sources and 3,888 Indic clips were preserved during the switch. The expanded Runpod CPU suite passed 188 tests and 23 subtests; these are correctness checks, not quality or RTF benchmarks. [Acquisition snapshot](expanded-acquisition-status.json).
