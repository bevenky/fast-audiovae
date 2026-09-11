# Independent trial queue

This first queue stopped at its predecessor check because the downstream trial failed to save its final checkpoint. No GRAIL or quiet model jobs were launched by it. The failed status remains unchanged. The [corrected queue](../independent-trials-v2/execution.md) now manages execution.

Original launch contract: the queue was launched on the existing Runpod. It waits for downstream-selection recovery to complete2000updates, then runs GRAIL initialization, GRAIL recovery2000, and quiet-preserving recovery2000, sequentially on one GPU. It does not retry failed jobs, extend budgets, choose a winning checkpoint, or launch another pruning cut.

- Source bundle SHA256 `31a0725424088ac1a8fc72781f52e84d4ac53549bc93fa0c6a30fc58aa6b79af`.
- Configuration SHA256 `9d39680a7c87a07266fcfff59a17f7dd7782bf2f7dc9180d3e075d390a4574b7`.
-75local and75qualified remote CPU tests passed. Remote log SHA256 `23c9230a09c7096b534736a753c8644fdfbced8e4f90e93da3db45baa9d1f396`.
- Remote root `/workspace/fast-audiovae-independent-trials-20260911-v1`; aggregate status in `status.json`.
- Source hashes, exclusive output/launch locks, exact predecessor identity, completion receipts and checkpoint bytes gate each next job. A technical failure leaves following jobs pending. Quality is evaluated at the declared endpoint without changing the method mid-run.
- TensorBoard port8888 switches only after the next candidate passes initial parity and starts real updates. Prior event files remain untouched. A dashboard error is recorded separately and does not cancel active training.
- The code-only aggregate collector independently verifies all90finite Adam states, matched24,000-source order and saved quality summaries. Its output excludes audio, latents, model values, source IDs and per-recording values.

[Fixed configuration](config.json) · [Complete experiment register](../experiment-register.md). Every run begins from the original teacher and declared calibration fits with fresh Adam/RNG. No trained student initialization, automatic promotion or new inference operations.
