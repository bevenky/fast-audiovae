# Fresh G plus startup execution

11 September 2026. Source SHA-256: `6bc3e3939af7276cfe7e393a0dd6bc1cf83bdcb6e8289de54fc88affd4c35152`. Six focused tests passed and two independent reviews found no execution blocker.

Runpod root: `/dev/shm/fast-audiovae-grail-startup-init-20260911-v1`. First launch stopped before computation because `grail_candidate_recovery` was absent from the launch path. Its log remains `initializer.log`. The corrected launch, `initializer-v2.log`, combines the frozen G and quiet-recovery dependency paths and is running. No source, model or data change was needed.

The initializer qualifies only if the original G baseline reproduces, the native constrained fit is feasible, all six calibration waveform startup checks pass, and original data/state guards remain intact. Completion alone is not qualification. Matched fresh64 pilots follow only after this review; the 2,000-update run has not started.

Completed successfully in initializer-v3.log after adding the existing aggregate reporting helper to the isolated source directory. Both import-only failures remain preserved. Native fit,6/6calibration,13/13development and all guards passed. See findings.md.
