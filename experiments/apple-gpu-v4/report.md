# GPU matrix repair

The repaired candidate passes the original numerical gates and retains a substantial short-run speedup. No tolerance, weight, input data or history definition was changed. This is bounded experimental qualification, not a sustained performance guarantee.

The trace located the first rounding difference at the first upsampler, stage 0. Its two separately rounded channel dot products differed slightly from the original FP32 convolution. Those differences propagated and grew through the unchanged residual operations and Snake activations. Eager execution also failed, so compilation was not required to trigger the problem. All same-input local transpose checks passed; the failure appeared after propagation. These comparisons establish agreement with the original FP32 implementation, not accuracy against an FP64 oracle.

The fix uses one paired current/previous-channel reduction at stage 0 and retains the original V3 two-projection matrix form at stages 1–5. The trace favored the paired form at stage 0 and the V3 form at every later stage. Applying the paired form everywhere was rejected after a late `Expressive0` state failure: two elements failed at `stage5.residual2.history`, maximum error 4.0054e-5. The interleaved and original-first-stage alternatives remain unused and were not GPU-tested. All failed receipts are retained.

Both repaired variants passed 60 waveform comparisons, including reference controls, and 1,014 direct state comparisons each: all 26 independently evolved histories after 39 nonempty calls. The original `atol=1e-5, rtol=1e-4` remained in force. Coverage includes three languages, two expressive fixtures, zero/quiet latents, mixed 40/80 ms packets, empty/flush, reset, changed-future prefixes, interleaved streams and input/state nonmutation. Seven-frame fixture outputs retain all 13,440 samples. History retains the original 26 tensors, 647,424 bytes.

| Qualified variant | Maximum waveform error vs eager GPU | Maximum waveform error vs original CPU full reference | Maximum state error vs eager GPU |
| --- | ---: | ---: | ---: |
| Hybrid transpose | 1.1995e-6 | 6.1095e-7 | 2.2739e-5 |
| Hybrid transpose + 19 pointwise matrices | 1.3337e-6 | 7.3761e-7 | 2.2531e-5 |

Errors above the absolute term passed the unchanged combined absolute/relative test. The maximum across all 60 waveform checks was 1.4603e-6, from an original eager-versus-CPU reference control. Each variant compiled two packet shapes; counters remained unchanged through qualification, with no graph breaks. Source and fixture checks passed.

The matched performance screen used three 960 ms speech excerpts, one qualification sweep, one warmup and two measured sweeps, with balanced arm ordering. Every arm used the same public GPU API, including input upload, all-history finite checks, synchronization and owned CPU-ready output. Construction and compilation were excluded; ordinary fresh-history initialization remained timed. Torch 2.14.0, FP32, one host thread, fast math and CPU fallback disabled.

| Arm | 40 ms pooled RTF | Reduction vs control | 80 ms pooled RTF | Reduction vs control |
| --- | ---: | ---: | ---: | ---: |
| Original compiled GPU | 0.125695 | Control | 0.073811 | Control |
| Hybrid transpose | 0.065322 | 48.03% | 0.037106 | 49.73% |
| Hybrid + pointwise matrices | 0.057877 | 53.95% | 0.032149 | 56.44% |

Both repairs won all six matched stream pairs at each packet size. For the combined repair, median paired reductions were 56.22% and 58.36%; these differ from the pooled ratios above. All 72 streams and six preparation waveform checks passed, worst error 3.5018e-7. Each stream emitted 46,080 samples: 24 packets at 40 ms or 12 at 80 ms, with no missing chunks or flush samples. The six compiled graphs and 2,824 captured calls remained unchanged through timing.

A separate short 80 ms paced check tempers the throughput result. Mean completed service time fell from 7.3323 to 6.0245 ms, a 17.84% reduction; its back-to-back control fell from 5.5052 to 2.8968 ms. Means exclude each stream's first packet. Each arm completed 16 paced calls with no deadline misses, but maximum service time was 9.446 versus 9.485 ms and maximum readiness delay from scheduled arrival was 17.446 versus 19.505 ms. Arrival lateness approached 10 ms. Tail latency did not improve in this sample, and the roughly 54–56% unpaced reduction must not be presented as a live paced guarantee. All 64 paced/back-to-back waveform checks passed, maximum error 1.6764e-8; compiler counters and sources were unchanged.

Evidence: [numerical analysis](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v4/numerical-analysis.md), [hybrid qualification](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v4/hybrid-r1.json), [combined qualification](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v4/hybrid-pointwise-r1.json), [matched screen](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v4/screen-repair-r1.json), [paced check](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/apple-gpu-v4/paced-repair-r1.json).
