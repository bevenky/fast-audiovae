# Completed 4500-to 5000 replay

All 500 requested updates completed in 117.82 seconds in an isolated directory. No update beyond 5000 ran, and the original checkpoints, source order and frozen teacher/body tensors were preserved. The replay was **not bitwise exact**, so its intermediate observations cannot be treated as an exact reconstruction of the historical training path. No exactness threshold was relaxed.

## What was reproduced and checked

- Restored the actual step 4500 group weights, AdamW moments, counters and Python/NumPy/Torch/CUDA random states.
- Consumed exactly fresh sources 10500 through 11999, in the original ordered 500 triplets. Singleton execution, accumulation 3, FP32, learning rate 0.00003, loss coefficients and parameter scope were unchanged.
- Compared every original branch loss at each update. Recorded real parameter displacement and its dot product with the existing minibatch gradient without an additional backward pass.
- Measured four fixed development cases at 4500 and every 25 updates through 5000. These cases never entered optimizer updates. Diagnostic state guards preserved weights, buffers, random state, modes, gradient slots and backend settings.
- Observed every teacher forward already called during these 500 updates. **All 1500 teacher waveforms matched the cached targets bitwise on every valid sample, with maximum error zero.** No additional teacher call was needed for that check.
- All 16 focused CPU tests passed locally and in the actual Runpod environment before launch.

## Strict replay result

The first discrepancy appeared at 4501. The total differed by 2.13e-10, waveform loss by 2.33e-10 and mel loss by 2.98e-8; feature loss matched. These initial differences are at floating-point rounding scale. Their origin was not isolated. Backend/workspace selection is a possibility, not a demonstrated cause.

| Final component | Difference from original 5000 |
|---|---:|
| Raw group parameter relative L2 | 0.011606% |
| Effective WN weights and other parameter relative L2 | 0.012100% |
| Largest raw parameter absolute difference | 0.000120862 |
| AdamW first-moment relative L2 | 2.7663% |
| AdamW second-moment relative L2 | 1.0475% |
| AdamW step counters | Exactly equal |
| RNG components | {'torch': True, 'cuda': True, 'python': True, 'numpy': True} |
| Optimizer settings, execution identity and source ledger | True |

Every complete checkpoint tensor/container was compared; torch.save archive bytes were not used as the equality criterion. The discrepancy summary also reconstructs effective weight-normalized tensors on CPU to separate effective weights from their g/v representation.

## Endpoint quality comparison

Initial 4500 MAEs passed the existing saved-metric checks for all four cases. Final 5000 MAEs failed those same checks: relative tolerance 1e-5 and absolute tolerance 1e-6.

| Source | Original 5000 MAE | Replay 5000 MAE | Relative difference |
|---|---:|---:|---:|
| Spanish | 0.003153970 | 0.003143110 | -0.3443% |
| Kannada | 0.002569167 | 0.002556935 | -0.4761% |
| Kashmiri | 0.017400223 | 0.017364399 | -0.2059% |
| Whistling | 0.000860731 | 0.000859159 | -0.1826% |

The endpoint differences are small relative to the observed 4500-to 5000 regressions, but they exceed the existing acceptance tolerance. This supports only a qualified qualitative interpretation of replay trajectories. No historical intermediate case scores exist at 25-step resolution to establish their exact counterpart values.

## Initial trajectory observations

All four amplitude trajectories move in both directions; they do not show a simple irreversible increase. The independent trajectory audit quantifies synchrony, sign reversals and source-batch associations. In this replay,33 of 500 AdamW displacements have a positive dot product with the current weighted minibatch gradient. Momentum can produce such steps; this count alone does not identify a bad learning rate, optimizer or batch size.

## Files

- completed.json: strict per-step and full endpoint comparisons, target checks and 21 snapshots.
- replay.jsonl: original loss comparisons, actual update directions and each source triplet's cached teacher level, quiet occupancy, language and dataset.
- case-trajectories.json: four fixed-source diagnostics at 21 checkpoints.
- discrepancy-summary.json: CPU tensor/moment comparisons and quantified endpoint differences.
- source-validation.json: frozen source hashes and 16-test validation receipts.

Remote output: `/tmp/fast-audiovae-replay 4500-5000-v 1`. The replayed checkpoint remains there; no checkpoint or audio was downloaded to the Mac.
