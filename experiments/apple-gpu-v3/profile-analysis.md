# Read-only GPU profile review

The saved measurements pass their stated accounting and matching-packet hash checks. They measure CPU-side submission/completion spans, not exclusive GPU layer durations. No GPU work was run for this review.

## Host profile

Means in milliseconds; all initial packets excluded exactly as in the source. The unprofiled means combine the before/after phases.

| Packet | Unprofiled model host | Combined finite check + wait | CPU-ready total |
|---|---:|---:|---:|
| 40 ms | 1.6924 | 2.6957 | 4.8133 |
| 80 ms | 1.7430 | 2.9019 | 5.0635 |

The instrumented packets each contain 45 convolution calls and 83/84 generated Metal dispatches at 40/80 ms. Their mean **host** contributions are:

| Family | Calls/packet | 40 ms host us | 80 ms host us |
|---|---:|---:|---:|
| depthwise convolution | 19 | 499.484 | 457.383 |
| final convolution | 1 | 31.599 | 23.925 |
| generated metal dispatch | 83 / 84 | 201.663 | 211.239 |
| pointwise convolution | 19 | 442.042 | 438.368 |
| transposed convolution | 6 | 170.901 | 173.233 |
| unhooked model host and hook bookkeeping | — | 490.993 | 439.584 |

Hook record metadata and append work lie outside each individual hook timer but inside the model span. The remainder therefore cannot be called pure Python overhead. Generated kernels combine operations; they are not all Snake kernels. Off/on/off total means are 4.9888/5.1151/5.1382 ms at 80 ms, and 5.0004/4.6368/4.6262 ms at 40 ms. This run does not isolate instrumentation overhead from drift.

## Queue and validation diagnostic

| Route | Pacing | Steady n | Mean total ms | Max all calls ms |
|---|---|---:|---:|---:|
| public | none | 14 | 6.7495 | 11.7142 |
| public | 80 ms | 14 | 12.1600 | 20.6254 |
| split_wait | none | 14 | 7.0074 | 9.4100 |
| split_wait | 80 ms | 14 | 11.6406 | 16.0295 |

After the added completion barrier, the finite-check span averages 0.5994 ms unpaced and 0.5144 ms paced; preceding completion waits average 3.7158 ms and 4.7739 ms. This supports separating completion wait from validation work. The extra barrier changes execution, so these are diagnostic spans, not a precise removable-overhead estimate.

The public route completes GPU work and returns CPU audio on every call. **The results do not show prior-packet backlog.** They show slower service after 80 ms pacing. All 32 public calls finish within 20.6254 ms of their measured start, but start lateness relative to scheduled arrivals was not logged. Two 640 ms bursts per condition cannot establish sustained deadline behavior or diagnose power/clock/scheduler causes. Unpaced public steady means vary 5.3357 → 8.1632 ms between repetitions.

## Accounting and evidence

- 54 profile packets, 18 instrumented; 8 prior warmup API calls. All saved host component sums equal total exactly.
- 64 queue packets plus 4 prior warmup calls; summary excludes the 8 initial packets. All corresponding packet byte hashes match across every route/repetition.
- Summary means were independently recomputed. No waveforms, latent values, new model calls or GPU activity were created.
- Neither snapshot independently pins all model/generated-code/weight/device identities or repeats original-ONNX state validation; link existing qualification provenance separately. The stopped Instruments capture supplies no GPU attribution.

Source boundaries: [profile hooks](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v3/profile_gpu.py:12), [finite-check and completion boundary](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v3/profile_gpu.py:49), [queue pacing and split barrier](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v3/queue_probe.py:22), [output equality and exclusions](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v3/queue_probe.py:42), and [production GPU boundary](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae-apple-gpu/src/fast_audiovae/gpu.py:135). Exact source/result hashes and full per-family/per-repetition aggregates are in [profile-analysis.json](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v3/profile-analysis.json).
