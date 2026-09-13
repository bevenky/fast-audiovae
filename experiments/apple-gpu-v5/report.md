# Direct upstream AudioVAE2 GPU comparison

This September 13, 2026 check uses the actual pinned OpenBMB VoxCPM2 AudioVAE class and original checkpoint on Apple M5 Max MPS, plus its official `streaming_decode()` implementation. It is independent of the original-weight streaming port used as the V4 reference. All 45 upstream weight-normalization hooks and sample-rate conditioning remain active. Source and checkpoint hashes were verified, checkpoint loading was strict, and parameters were asserted FP32 on MPS. CPU fallback and fast math were disabled.

The repaired candidate is V4 `hybrid_pointwise`: paired matrix projection in the first upsampler, V3 projections in the later five, and 19 pointwise matrix replacements. Original weights and causal histories are retained.

| Decoder | 40 ms streaming RTF | 80 ms streaming RTF |
|---|---:|---:|
| Upstream AudioVAE2 on GPU | 0.16657 | 0.09148 |
| Repaired GPU candidate | 0.05832 | 0.03020 |
| Decoder-time reduction | 64.99% | 66.98% |

These are matched incremental streaming calls made back-to-back, not batch decoding. Both return owned CPU-ready waveform arrays, including upload, GPU completion and finite checks of audio and all 26 histories. The upstream adapter does not require contiguous owned histories because the official implementation retains views. Testing used one host thread, FP32, three 960 ms speech crops, one warmup sweep and two measured sweeps with alternating arm order. The upstream baseline includes weight-normalization recomputation; the optimized candidate uses equivalent frozen exported coefficients. Model construction and compilation were excluded. Both 40/80 ms graphs were prepared once; no later compilation occurred. These short results do not establish sustained paced-arrival performance.

All 80 waveform comparisons passed the unchanged atol=1e-5, rtol=1e-4: two initial comparisons, 28 candidate-versus-upstream streaming packets, seven upstream-stream/full checks, seven candidate/upstream-full checks and 36 timing-stream checks. Maximum absolute waveform difference was 1.21445e-6. Coverage included Bengali, English, Spanish, two expressive fixtures, zero and attenuated latents, and mixed 40/80 ms packets. Exact emitted sample counts passed. This is numerical waveform qualification, not a new PESQ/STOI/MOS evaluation.

Both paths finite-check their internal GPU states here, but this run does not numerically compare upstream state tensors against candidate states. The earlier 1,014 numerical state comparisons apply to the original-weight eager GPU port reference, not this upstream-class run.

Recorded ordinary streaming API time was 3.2794 seconds. Ten fixed full-upstream reference calls, model construction and initial compilation are outside that counter. Historical source-provenance metadata originated from a CPU benchmark; this run independently records and asserts MPS execution. The source, runtime and checkpoint were not modified.

The candidate remains an experimental GPU implementation awaiting public-loader integration. CPU remains the default. The prior V4 80 ms paced result remains separate: repaired service RTF 0.07531, versus its compiled-GPU control 0.09165; that was not a direct paced comparison against the upstream class.

Evidence: [receipt](upstream-gpu-r1.json), [script](upstream_gpu.py), [log](upstream-gpu-r1.log). Published source: [AudioVAE2 at pinned OpenBMB revision](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py). Model: [VoxCPM2 pinned checkpoint](https://huggingface.co/openbmb/VoxCPM2/tree/32279effe8c19989596f05d353d1447f51d9e915).
