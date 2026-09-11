# Reconstruction initialization execution

The approved two-variant experiment completed on Runpod. No optimizer updates, new inference modules, checkpoint promotion, commits or push occurred.

- Run root: /tmp/fast-audiovae-progressive-pruning-v1/reconstruction-init-v1
- PID: 1083755, exited after completing both variants.
- Core experiment elapsed time: 46.635 seconds, excluding setup, upload, tests and initial data/coverage loading.
- Runtime: PyTorch 2.14.0+cu126, cuDNN 92501, FP32 model operations, TF32 disabled, qualified singleton inference.
- Calibration: 72 sources, all valid feature rows; development: 96 disjoint sources.
- Local and qualified remote CPU tests: 9 passed.
- Source freeze SHA256: be5d0044ff3f0d4f4171b220dc740bfd9460fe590a109a2de9ac7a346da7f516
- Main source SHA256: de5362361b1dfd387c9506fefa93e13435d772f784fcc344390791265efa1913
- Test source SHA256: 0eaa15c493988119d526eb702565866a0be9de17fc188cb9c330a604fa34fbc0
- Remote CPU test log SHA256: 0f9c3b52c9360992913efd791a3f43d88558de91e99a51e28846aa35578ee86b
- Preserved step0 SHA256: bd9a09c1ab5e86fce8f5ba1f435565c9dfff76b83c32d413d037a54d0df5942f
- Preserved step5000 SHA256: 4dd64e0d165ab23aa56e5c9f0e0fe0d281e510dfb3db8d6cd863a2be0640fe47

Original step0 full-panel metrics reproduced. Every fitted native operation and effective-weight writeback passed the unchanged numerical tolerance. Protected source/data/checkpoint files, original teacher state, original step0 state and unselected/frozen candidate tensors were preserved.

Fitted operator artifacts remain on Runpod. No audio, latent arrays, source identities or per-recording metrics were transferred. The locally saved aggregate-results.json and aggregate-comparisons.json contain aggregate metrics and counts only. Comparisons against the preserved step5000 report required no additional model forwards.

The immutable source bundle and plan were not changed after launch. Training and the old training monitor remain paused.

