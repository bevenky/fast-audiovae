# AMD precision evidence

The final AOCL four-worker screen passed the speed objective. Full 60-clip runtime checks and fresh quality scoring completed without errors. INT8 has small measured quality declines; the public runtime is unchanged.

- `screen-r1-results.json`: rejected original oneMKL two-worker port.
- `screen-aocl4-r1-results.json` and `screen-aocl4-r1-statistics.json`: single final matched screen and timing-noise analysis.
- `checks-aocl-r2.json`, `checks-schedule4-r1.json`: exact integer and stage/scheduling validation.
- `quality-aocl4-r1-results.json`, `quality-identity.json`: full 60-clip execution and waveform identity, separate from perceptual quality.
- `quality/`: fresh scores, all 12 paired metrics, per-language differences, model/source pins and preflight records.
- `profile-*-results.json`, `diagnostic-summary.json`: separate diagnostic profiles. Summed worker times overlap.
- `builds/`, `dependencies/`: original build configuration, source/header/library pins and normalized records.
- `publication.json`: unchanged source hashes and original/published evidence hashes.

Private paths were normalized. Numerical observations and artifact digests were preserved. Model files, audio, weights and libraries are not distributed. Saved configs are historical audit records; use the [parameterized source tools](../../experiments/amd-precision/README.md) with trusted local inputs for a new build.
