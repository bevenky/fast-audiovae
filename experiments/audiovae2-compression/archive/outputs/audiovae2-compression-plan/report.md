# AudioVAE2 group compression audit

The [implementation plan](../../work/fast-audiovae/docs/audiovae2-compression.md) is on the new `audiovae2-compression` branch, created from `main` at `7150ae9bc4aff725b303c33b2417bd50d58a4f71`. No commit, push, model training or inference benchmark was performed during this audit. Existing candidates and uncommitted work were preserved.

The intended objective is complete-group distillation: the smaller group receives the same input/history as the original teacher group and jointly learns its final output. Individual hidden layers need not match. Keep the original teacher's final waveform as a second target through a differentiable frozen suffix.

The first candidate replaces stages 2–4 as a single box with the same 1,024-channel 200 Hz input and 128-channel 12 kHz output. Its internal widths become 256, 128 and 128 instead of 512, 256 and 128. Keep all nine residual units initially. The static count falls from 8.9912 to 5.1741 GMAC/s, a 42.45% reduction with unchanged temporal support. A later six-unit version reaches 4.7823 GMAC/s, a 46.81% reduction, while shortening support. These are arithmetic estimates, not measured RTF or quality.

## Evidence

- [Exact architecture, group boundaries and cost analysis](teacher-structure-and-budget.md)
- [Static arithmetic and provenance](static-budget.json)
- [Reproducible analysis script](../../work/audiovae2-compression-audit/static_budget.py)
- [Method review and primary research](method-audit.md)
- [Quality, numerical and streaming qualification](validation-audit.md)
- [Live Runpod corpus and storage inventory](dataset-audit.md)
- [Corpus metadata receipts](dataset-audit.json)

The synthesis chooses a wider replacement boundary than the initial method review's single-stack diagnostic because the latter alone offers insufficient arithmetic savings for the requested CPU objective. The width-first choice also separates the large matrix reduction from the smaller additional benefit and context loss of deleting residual units.

Live metadata found 510.927 train-split hours; known reserved source/hash/parent exclusions leave 501.698 hours. All 22 Indic languages remain represented. The H100 was idle, but persistent storage had only about 182 MiB free. Final manifests, verified checkpoint space, initialization and calibration are preflight work before launching the new pilot.
