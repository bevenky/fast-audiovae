# Fresh optimizer comparison

Approved on 11 September 2026. Preparation is underway; this page will link the live launch receipt once training starts. The completed G-plus-startup run and every earlier experiment remain preserved.

The question is whether a different optimizer can recover the same pruned decoder faster while keeping the existing startup protections. Architecture, teacher, data order, loss weights and the final acceptance rule stay fixed. None of these changes adds CPU inference operations.

| Arm | Nine pointwise direction matrices | Other 81 trainable tensors |
|---|---|---|
| AdamW | AdamW | AdamW, existing settings |
| Muon + AdamW | Declared Muon variant | AdamW, existing settings |
| NorMuon + AdamW | Declared NorMuon variant | AdamW, existing settings |
| Shampoo + AdamW | Two-sided Shampoo with Adam-style norm grafting | AdamW, existing settings |

The [formula and source record](optimizer-sources.md) specifies the implementations and their differences from library defaults. Only the nine square residual pointwise matrices at widths 384, 256 and 128 receive a matrix optimizer. Weight-normalization gains, Snake parameters, depthwise kernels, transposed convolutions and biases remain on AdamW.

## Two stages

1. **Matched qualification:** three learning rates per arm, each for 64 updates from the same fresh initializer and empty optimizer. The first AdamW trial must reproduce all original non-timing update scalars from the completed corrected64 reference. Stop on a mismatch. Each trial uses the same 768 ordinary sources and six separately counted recurring calibration anchors.
2. **Independent recovery:** run each qualified arm for 2,000 updates, starting again from the original teacher-derived initializer. Never resume the qualification endpoint or the previous trained model. The same 24,000 ordinary sources occur once in each independent run. Select learning rates using a fixed, separate 12-source calibration probe, with no development-based selection.

AdamW and Shampoo pointwise learning rates are 0.000015, 0.00003 and 0.00006. Muon and NorMuon rates are 0.000075, 0.00015 and 0.0003, with direction RMS 0.2. All complementary AdamW tensors keep learning rate 0.00003, betas (0.9, 0.99), epsilon 1e-8 and zero weight decay. These small grids test viable settings; they do not establish globally optimal hyperparameters.

The qualification winner for each arm is its lowest finite final calibration-probe objective among trials that pass state, source, preservation and useful-movement checks. Ties use the lower matrix learning rate, then the candidate name. Development metrics are reported but never used to select rates. Shampoo must exercise its inverse-root refresh within the pilot; NorMuon must hold independent per-output-row states.

## What remains fixed and visible

All 90 group tensors train jointly. The frozen teacher, encoder and outer decoder stages stay unchanged. We retain physical batch size one with 12 accumulated examples, FP32 computation, TF32 off and the established deterministic execution path. The existing 12 physical startup constraints and bounded correction are evaluated against the actual mixed-optimizer proposal. A changed optimizer counter alone does not count as learning; zero weight displacements are recorded and disqualify a short pilot.

Losses and update timing appear each step. The fixed 96-source development panel is measured at 0 and 64 during qualification, then 0, 250, 500, 1,000, 1,500 and 2,000 during recovery. Report waveform error and correlation, mel and feature error, active and whistling amplitude, overshoot, startup passing and all seven overlapping quiet cohorts. Correlation is not a percentage of perceptual accuracy.

Dedicated investigation of the quiet-window regression begins only after all four optimizer comparisons, as requested. Routine quiet measurements continue during these runs. No pruning change or model promotion is part of this experiment.

## Reproducibility and space

Each run records source/configuration hashes, fresh initialization and RNG hashes, data-prefix hashes, genuine optimizer activity, startup corrections and preservation checks. Save full states at 0, 1,000 and 2,000 and group-only snapshots at 500 and 1,500 for the later quiet audit. Keep all prior artifacts. Check free space before each run and snapshot; stop safely if there is not enough room. The Runpod has about 1.38 GB free in shared memory at preparation time, so new audio downloads are unnecessary and are not scheduled.

Audio, latent tensors, model values, raw source identifiers and per-window results remain on Runpod. Only aggregate statistics and provenance hashes are copied locally. The existing TensorBoard URL will be reused with fresh comparison events, while old events remain available on disk.
