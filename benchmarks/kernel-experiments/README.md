# CPU kernel measurements

[Results](../../docs/cpu-kernel-results.md) summarize the matched Intel and AMD decoder campaigns. [results.json](results.json) includes completed screens, per-clip changes, profiles and correctness evidence. Apple timings are explicitly excluded from promotion because of drift.

Each final campaign validates 60 recordings, then times ten fixed clips with five repetitions and two warmups. All models use CPUExecutionProvider, FP32 and ONNX Runtime 1.29.0. Timed calls use fresh causal history and exclude encoding, loading, warmup and separate profiling. The comparison uses AudioVAE2's 48 kHz decoder and Pocket continuous Mimi's 24 kHz decoder.

The `amd/` and `intel/` directories retain all 250 final timing observations, all 331 validation gates, frozen configurations and candidate manifests for each host. These are measurement records, not ready-to-run configurations for another filesystem. [publication.json](publication.json) records original and published hashes; machine-specific paths use workspace aliases, and numeric measurements are unchanged.

The [experimental source](../../experiments/cpu-stage) contains stage, matrix and composition tools. Use the [comparison harness](../compare_decoders.py) with locally prepared models and case archives to reproduce a campaign. Model, library, corpus and harness hashes are recorded in the evidence. No weights or recordings are included here.

The separate [CPU profiler](../profile_candidate.py) validates one explicit real latent shape and excludes its two warmup intervals from three-call operator aggregates. It requires a frozen configuration hash and explicit CPU affinity. Profiling durations locate bottlenecks; they do not replace the unprofiled RTF measurements.
