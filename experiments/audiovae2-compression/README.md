# AudioVAE2 group compression

Research snapshot archived on 11 September 2026. The descriptions below record successive experiments, not a currently running job or a production replacement. See the [archive index](../../ARCHIVE.md) for the compression method, experiment register and measured outcomes.

Experimental branch: `audiovae2-compression`.

Replace stages 2–4 with a smaller jointly trained group while preserving its input/output interface. Use the original AudioVAE2 encoder and decoder as the frozen reference. The first candidate narrows internal channels and retains all nine residual units; reducing them to six is a later option if needed.

The [plan](../../docs/audiovae2-compression.md) covers initialization, whole-group distillation, data reuse, streaming correctness and Intel-first CPU qualification. The static estimate is 42.45% fewer decoder MACs. Quality and runtime gains have not been measured.

The model, training runner and TensorBoard reporting are implemented. The pilot uses the same frozen stage-1 input for both paths and directly matches the complete stage-4 output, together with teacher-waveform and multiscale mel losses. Intermediate stage-2/3 measurements are diagnostic only because their widths change.

Numerical, export and streaming checks must pass before fitting. The first run uses 1,000 optimizer updates across 3,000 distinct sources, with separate calibration and development data. It targets 0.99 nonquiet waveform correlation while also tracking quiet residuals, peaks, languages and expressive sounds. This is a reconstruction target, not a claim of 99% perceptual quality.

Before that continuation, the [settings comparison](../../docs/audiovae2-settings-screen.md) tests two mel definitions and two learning rates from the same initialization. It records which settings come from author disclosures and which are specific to frozen-block distillation.
