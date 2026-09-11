# AudioVAE2 research archive

Start with the [compression method and measured learnings](experiments/audiovae2-compression/archive/outputs/audiovae2-compression-preflight/progressive-pruning-v1/method.html) and the [experiment register](experiments/audiovae2-compression/archive/outputs/audiovae2-compression-preflight/progressive-pruning-v1/experiment-register.md).

The archive preserves successful, failed, superseded and paused experiments. Historical findings are dated observations, not claims about a currently running process or a production model. The current implementation sources remain in [ConvNeXt](experiments/convnext/) and [AudioVAE2 compression](experiments/audiovae2-compression/).

External helper code and report trees are mirrored below `experiments/{convnext,audiovae2-compression}/archive/`, retaining their original `work/` and `outputs/` paths. Source-only contents of historical deployment bundles are under `archive/source-bundles/`. The original bundle hashes and member hashes are recorded in the selection manifest.

Reports that contain raw numerical payloads or large individual-window collections are labelled aggregate projections. They include the original report hash and omitted field paths. Configurations and scalar metric metadata are retained where payload-free. The manifest records excluded files; it does not present an excluded or projected report as an exact archival copy.

Audio, latents, model weights, ONNX binaries, caches, environments, TensorBoard events and raw logs remain outside Git. Archiving does not modify the original artifacts. On 11 September 2026, the project owner authorized removing obsolete AudioVAE process state and model artifacts from Runpod while retaining every audio dataset. Git preserves source and findings, not the deleted checkpoints; reproduction requires rebuilding those assets using the recorded configurations and runtime.

The last inspected local pause receipt, dated 2026-09-11 16:04:03 UTC, records AdamW complete and audited, Muon paused with 253 logged updates in process memory, and NorMuon/Shampoo deferred. Only Muon's step-zero checkpoint was on disk in that receipt. This archive is not a checkpoint of the paused process and makes no claim that the remote process still exists.

The [selection manifest](experiments/audiovae2-compression/archive/selection-manifest.json) records exact copies, projections and exclusions. The [archive preparation tool](experiments/audiovae2-compression/archive/prepare_archive.py) records how files were selected.

The later [process cleanup receipt](experiments/audiovae2-compression/archive/outputs/audiovae2-compression-preflight/progressive-pruning-v1/optimizer-comparison-v1/user-authorized-memory-discard-20260911T175026Z.json) records the authorized discard of the paused Muon process at 17:50 UTC. No new checkpoint was produced; the last journaled update was 253.
