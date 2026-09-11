# Canonical heldout targets

This creates a separate evaluation appendix for the frozen AudioVAE2 encoder's verified execution path. It preserves the historical paired panel and all checkpoints. The existing training pool remains the declared cached-latent distribution for the matched learning-rate experiment.

`render_canonical_targets.py --output /tmp/<new-inventory-directory>` checks authentic sources without running a model. Add `--render`, using another new directory, to inventory first and then create the new cache. Both commands need the existing diagnostic, natural-history, and frozen student source directories on `PYTHONPATH`.

The registry uses parent-checkpoint heldout rows, the verified validation-appendix and language-validation manifests, and the pinned `corrected-screen/addon-validation.json`. The last contains the additional `freesound:266716` Yell source that the older registry omitted. Each natural source's original bytes must match its declared SHA-256 and prepared mono 16 kHz duration. No cropped reference substitutes for a missing natural recording. The three synthetic fixtures use their exact full six-second historical reference tensors, without regenerating randomness or treating zero latents as zero input.

Rendering encodes each complete source in a singleton batch with cuDNN disabled only during encoding. It restores the prior backend setting, then performs normal singleton teacher decoding over all source latents. The renderer copies historical crop starts, context, score lengths and valid samples, slices full-source targets, and retains the six-sample interior exclusion. Neither encoder nor decoder weights change. The new cache key records the source input and both generated tensor hashes.

`heldout.pt` and `receipt.json` retain all 285 crops and 147 sources, including 144 natural sources and three fixtures. Quiet-window counts are newly computed from the teacher and sealed in the receipt; they may differ from the old panel's counts. Source receipts report latent and waveform differences from the old cache, including latent channel 63 separately. Inventory errors stop rendering before any model call. Outputs require a fresh directory under `/tmp` and cannot overwrite the old panel.

`canonical_evaluation.py` provides:

- `load_canonical_cache(receipt_path)` returns `(heldout, metadata, receipt)` after byte and tensor-identity checks.
- `build_canonical_evaluation(engine, heldout, metadata, receipt)` uses the existing evaluator once, with peak-energy capture during the same forwards.
- `compare_canonical_reports(before, control, candidate)` applies the unchanged pilot thresholds to the separately identified canonical domain.

The learning-rate candidate must pass both the historical and canonical screens before continuation is considered. A pass is a bounded continuation decision, not a claim of quality equivalence or deployment readiness. Missing or zero-denominator required metrics fail instead of passing silently. Small language/event groups remain visible without unsupported population claims.

Local validation so far: Python syntax compilation. Source coverage, rendering, and GPU numerical parity require the explicit Runpod execution; this helper itself has not run a model locally.
