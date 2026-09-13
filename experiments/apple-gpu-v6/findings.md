# Integrated public GPU comparison

Both single invocations passed on the same 16 package Python sources. FP32 MPS, one host thread, no CPU fallback or fast math. The fast arm was constructed with `fast_audiovae.load(device="gpu")`; the reference used the actual pinned upstream AudioVAE class, original checkpoint, official streaming API, default 48 kHz output conditioning and all 45 weight-normalization hooks.

| Decoder | 40 ms RTF | 80 ms RTF | 80 ms mean completed packet |
|---|---:|---:|---:|
| Actual upstream AudioVAE2 MPS | 0.169588 | 0.093286 | 7.463 ms |
| Integrated public fast AudioVAE2 MPS | 0.057703 | 0.032055 | 2.564 ms |
| Pocket Mimi MPS | Unsupported | 0.063438 | 5.073 ms |

Each table cell uses the same three aligned 960 ms speech crops, one warmup and two measured rounds, six streams per cell. All samples and both rounds are retained. Public AudioVAE2 reduced pooled time versus actual upstream by 65.97% at 40 ms and 65.64% at 80 ms, winning all six matched pairs at both sizes. Its 80 ms pooled time was 49.47% lower than Mimi's in this short test. AudioVAE2 emits 48 kHz; Mimi emits 24 kHz. These are completed CPU-output timings, including upload, synchronization, input/output checks and flush. Both Audio paths check all histories; Mimi checks input/audio but does not perform the same all-history reduction. The upstream empty flush is explicitly a host shim because its official API has no flush operation.

The numerical comparison passed all 71 waveform checks at unchanged `atol=1e-5, rtol=1e-4`, maximum absolute difference 5.141e-7. This includes public batch lengths 3/7, carried streaming lengths 3+13, and one natural 200-frame English continuation (8 seconds) at each packet size for both Audio arms. The latter emitted exactly 384,000 samples per stream and the public output differed from actual upstream full MPS by at most 2.981e-8. Continuation results do not enter the RTF table.

The independent unchanged state panel passed all 1,014 comparisons (39 per each of 26 states), plus 60 waveform checks, reset/interleaving/prefix/empty and pre-call state immutability checks. Its state reference is the original eager literal-ONNX-weight MPS port, not the upstream Python class; actual upstream waveform comparisons are provided by the separate comparison above. Maximum state absolute difference was 2.2531e-5 and passed the original elementwise absolute-plus-relative tolerance. Exactly two compiled graphs remained; arbitrary lengths used eager MPS and no recompilation occurred.

Accounting: comparison 1,690 ordinary calls, 10.5266 seconds, plus two public-load preparation calls; state panel 93 calls plus two public-load preparations, with 0.5976 seconds of ordinary GPU work and state inspection. Public streaming load took 5.9532 seconds in the comparison (verification, construction and compilation combined, excluded from RTF), and 2.4003 seconds in the later qualification. This is a short matched benchmark and finite qualification panel, not a corpus/perceptual evaluation or a guarantee of all-machine latency.

Artifacts: `compare-r1.json`, `public-qualification-r1.json`, both logs and the source scripts are preserved. `accounting-audit.json` independently recomputes pooled RTF, call/sample/state counts and source identity from saved results. No additional runtime calls were made for that audit.
