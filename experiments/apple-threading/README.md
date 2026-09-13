# Apple worker scheduling experiments

These are retained experiments, not automatic runtime selections. CPU streaming with one thread remains the default. The current release payload also remains unchanged.

| Experiment | Short 80 ms result against its own control | Decision |
| --- | --- | --- |
| v1: matrix panels and depthwise/Snake channels | Approximately 12.5% less decoder time | Retain opt-in source |
| v2: shared workers for the first projection pair | Approximately 9% less decoder time | Retain opt-in source |
| v3: shared workers for the second projection pair | 4.01% more decoder time, slower in all 12 pairs | Reject; source archived |

The percentages come from separate matched experiments and must not be compounded into a claimed total. At 40 ms, gains remained inconclusive. Every experiment retains its numerical and state checks in its report and JSON records.

`archive/outputs/` preserves the scripts, reports and build receipts. `archive/files.json` records exact file hashes. Binaries, weights, audio and generated arrays are excluded. The v3 source is archived; it was restored out of the package after the failed speed test. The retained optional native APIs are in `native/apple/streaming/`, but the production graph does not enable them.

These scripts capture the original workspace layout and are not standalone package examples. To replay them, restore `archive/outputs/` under a workspace root, put this repository at `work/fast-audiovae-streaming-baseline`, and supply the pinned fixtures described in `config-template.json`. Restore the archived revision/source hashes required by each build receipt. V2 depends on the v1 build, and v3 depends on both. The v1/v2 source patches are cumulative against their original baseline, not patches to apply one after the other.

Final source validation before publication: 314 tests and 1,959 subtests passed. No additional performance run was needed for the archive.
