# Apple GPU experiment evidence

Short 40/80 ms CPU-ready streaming comparisons from September 13, 2026. The preferred candidate compiles the original GPU history implementation. CPU remains the default; these candidates have not been merged or selected by the public loader.

Read [the results](../../docs/apple-gpu-experiments.md) for the comparison and qualification scope.

The Python files are verbatim research-workspace snapshots. They were run from `outputs/apple-gpu-v2` alongside existing model exports, input manifests and `outputs/apple-gpu-v1`; this archive is not a standalone benchmark installer. Receipts include those source identities and settings. No audio, weights or latent values are included.

`combined-r1.json` and `combined-r2.json` are failed history-layout attempts. `combined-r3.json`, `qualification-r1.json` and `final-compare-r1.json` are the completed results after the layout fix. All attempts remain here.
