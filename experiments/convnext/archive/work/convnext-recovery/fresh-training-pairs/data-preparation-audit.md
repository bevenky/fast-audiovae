# Fresh training pair review

The generator preserves the intended causal contract: authenticate the complete supplied source segment, encode it without resetting at a scored crop, decode the same fresh latents continuously, then extract the original indexed windows. The actual padded cuDNN 9.25.1 encoder batch is checked row by row against an independent full-source singleton encode with cuDNN disabled. Accepted target pairs use the singleton reference latents and their own decoder output.

No model, GPU, training, or remote operation was run for this review. The generator remains a candidate until its actual-source qualification completes.

## Guard changes and checks

`prepare_fresh_pairs.py:51` now rejects negative or inconsistent absolute frame coordinates and any scored interval ending beyond three times the authentic 16 kHz source length. Storage may still contain the partial final frame and right padding, but those samples cannot become scored audio. The existing crop validator enforces the declared context-plus-score shapes and integer sample ratio. The historical mask still excludes the initial six scored samples for an interior reset; this review does not change masks or context length.

`prepare_fresh_pairs.py:66` validates FP32 shape and finiteness for both encoder paths and decoder outputs. The source plan and historical receipt are checked again after generation. The generation record includes the exact generator source hash and explicitly describes selection as an ordered crop prefix, with source count reported separately.

Sixteen CPU helper checks passed. They cover a partial final frame, an all-zero reference that could conceal an invalid scored tail, absolute context coordinates, compact independent tensor storage, invalid geometry, teacher dtype/shape/nonfinite output, and a localized channel error that passes pooled RMS but must fail the per-element qualification. Evidence is `helper-verification.json`.

## Scope and storage

`--pool gradient_calibration:32` selects the first 32 crop entries, not necessarily 32 distinct sources. That matches the current diagnostic consumer. Preserve this distinction in reports. Full continuation needs only `targeted_generator:12800`, since both arms resume existing optimizer, discriminator, EMA and balancing state. There is no new warmup or regular-generator run.

For a 93-frame crop, FP32 latents, waveform and original reference occupy `4 * 93 * (64 + 1920 + 640) = 976128` bytes. A 12800-crop overlay is at most 11.6364 GiB at that geometry, excluding small metadata. It cannot fit the original 6 GiB temporary-disk budget. The parent's verified 56 GB free `/dev/shm` resolves capacity without deleting or modifying the historical 15.5 GB cache.

The window helper explicitly clones compact storage, so saving a crop cannot accidentally serialize its entire source tensor. The sealing helper currently uses `dataclasses.asdict`, which deep-copies tensors and can add approximately one overlay's host-memory footprint during serialization. Check actual available RAM as well as the tmpfs quota. Retain capacity for both new checkpoints and an atomic replacement. Transfer the final manifests, results and chosen checkpoints to durable storage before the temporary environment disappears.

## Two-arm provenance

The updated `run_update_experiment.py` now loads fresh training through a separate receipt, retains the old pool identity as selection provenance, and binds the actual fresh pool identity into the experiment and checkpoints. A fresh training overlay requires the canonical evaluation panel. Both historical and canonical evaluation gates remain independent. Both arms consume the same ordered 12800 windows and discriminator views from the same sealed overlay.

The release gate should bind the exact fresh receipt, contract and payload hashes, plus the parent checkpoint, original selection plan and canonical evaluation identities. Merely setting `requires_fresh_training_pairs=true` is weaker than binding the already prepared payload. The target-domain migration is shared by both arms; learning rate remains the sole difference between the arms. Small intermediate health probes still use the historical diagnostic panel, while full canonical evaluation supplies the corrected-domain gate.

`audit_encoder_sample.py` and `validate_conditional_targets.py` serve different purposes. Their old-latent-to-teacher checks audit historical conditional pairs and cannot produce corrected encoder pairs. Old numerical failures remain separate evidence; they are neither silently overwritten nor used to relabel the fresh cache.

Reviewed generator SHA256: `0bc5daa8c33d78d1193523d0ef14dbe32b3542c056edb656f4d636d1c4693bc8`.
