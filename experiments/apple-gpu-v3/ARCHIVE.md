# Apple GPU matrix and queue experiments

See [report.md](report.md) for the results and their limits. CPU remains the default; no candidate is promoted by this archive.

The combined matrix screen reduced back-to-back latency by about 51% at both packet sizes. Its separate paced 80 ms mean improved by 32%. Waveform comparisons passed, but the unchanged internal-state qualification failed. The failed results and the aggregate diagnostic are retained alongside the successful timings.

`manifest.json` verifies each copied source, report and receipt. The scripts are workspace snapshots that reuse the V2 harness and local verified model assets. The V2 sources are preserved in the sibling `apple-gpu-v2` archive; those model and audio assets are not included here. Run only one GPU test at a time.

No audio, trained tensors, raw latent/state values or compiler-cache binaries are included. The failed Instruments capture log is included; its unusable partial trace stays in the workspace.
