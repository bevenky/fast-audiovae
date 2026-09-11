# Cached pilot data and storage readiness

The metadata selection is ready. No audio was decoded, teacher or student executed, new data downloaded, or remote file deleted for this preparation.

| Split | Distinct sources | Scored audio | Purpose |
|---|---:|---:|---|
| Calibration | 72 | 2.87 minutes | Channel initialization and fixed loss calibration |
| Development | 96 | 3.91 minutes | Fixed recovery diagnostics |
| Fitting | 3,000 | 123.10 minutes | 1,000 updates at batch size 3, each source used once |

All three splits contain all 22 scheduled Indic languages, English, Spanish, Portuguese, Mandarin, Japanese, French and Arabic. They each include the 11 reviewed expressive source labels plus previously input-qualified quiet windows and transitions. The fitting split has 511 expressive crops, accounting for 13.73% of scored duration, and 89 quiet / 97 transition crops. These are debugging subsets of the existing corpus, not a proposal for the final training mixture.

The 12,800-crop sealed cache contains 3,621 distinct sources, so it cannot supply 1,000 batch-eight updates without repeating sources. If preflight chooses a different batch size, either shorten the no-repeat fitting schedule or prepare additional pairs from the already downloaded corpus. Do not recycle these fitting indices silently. Source audio available to the fitting records totals 8.67 hours, but only the selected 123.10 minutes are scored; these exposure measures must not be confused.

## Runner interface

Read [pilot-selection-v1.json](pilot-selection-v1.json). `splits.calibration.indices`, `splits.development.indices` and `splits.fit.indices` index the existing `pairs.pt["pools"]["targeted_generator"]` list. Matching `rows` include cache keys, source hashes, parent identities, source manifests, tensor shapes, label provenance and crop coordinates. Fitting order is already deterministic.

- Cache: `/dev/shm/fast-audiovae-recovery-fresh12800-v1/pairs.pt`
- Receipt: `/dev/shm/fast-audiovae-recovery-fresh12800-v1/receipt.json`
- Expected cache SHA256: `1125b32246ebd102ef4a4c3ce9feca878c18b73a985afa77ee16ee5226df4f23`
- Selection identity: `0df9b99bcc2be99406368a35c8014272f6c525d0db29e6fe1396f98ccb80c483`
- Selection file SHA256: `b76341240028a0371b9acbb553fa5c1bd6714002da10d7900aa501cd998c3627`

The cached teacher pair metadata was memory-mapped on CPU without reading tensor values or executing a model. All stored tensors are FP32. Every selected nonstartup crop has 30 latent frames of context; startup crops begin at source frame zero. This exceeds the original decoder's 20-frame maximum support. Score exactly `[context_frames*1920 : context_frames*1920+valid_scored_samples]`. The earlier student's six-sample omission does not apply to this decoder.

Source IDs, recorded audio hashes and parent recording IDs are disjoint across all three splits and from the sealed canonical development panel. Deterministic regeneration produced identical JSON, and all 3,168 indices are unique. [Metadata checks](data-selection-checks.json). The generator is [data_setup.py](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/audiovae2-compression-preflight/data_setup.py).

Expressive labels describe recordings, not timed semantic events in every crop. Quiet and transition flags are historical input-energy measurements, not newly verified teacher-output properties. Speaker separation is not claimed for this debugging panel. One source per language/category in a small development panel establishes presence, not a robust population estimate. A fresh final holdout remains necessary.

Before model use, verify the full cache byte hash, source PCM authentication, copied-model parity and full-context group/teacher-target agreement under the preserved backend. Metadata readiness alone does not satisfy those numerical gates.

A subsequent byte-only check verified **all 3,168 selected source files**, with zero failures: 72 calibration, 96 development and 3,000 fitting files. Eight parallel workers read 1,630,169,194 bytes and compared SHA256 against the pinned source rows. Files remained stable during reading. This completes file-byte authentication; audio decoding, cached tensor values and model numerical checks were outside its scope. [Authentication receipt](source-authentication.json), [verification helper checks](source-authenticator-checks.json).

## Storage proposal for review

Persistent `/workspace` free space was 190,767,104 bytes. The following old generated runtime directory contains 18 ordinary single-link `.onnx`/`.json` files occupying 1,832,873,984 allocated bytes, approximately 1.71 GiB:

`/workspace/fast-audiovae-installed-wheel-20260908-r1/cache/build/179cca35cd01180fd1a4c59c0acd8afccb792918468d998ee035db301ef80beb/.build/automatic-graphs`

No process had an open file descriptor or memory mapping into that directory at inspection. Its sibling original exports, completed bundles, prebuilt artifacts, build source and receipts remain available. This is a proposed removal of reproducible installed-wheel test output. Preserve all corpus audio, teacher weights, retained checkpoints, optimizer states and sealed training data. [Exact proposal](storage-cleanup-proposal.json).

No deletion has been performed. Recheck available durable space after any approved cleanup and reserve space for the actual checkpoint/optimizer footprint before launch. The 11.4 GB pair cache is in volatile shared memory, which must not be treated as durable checkpoint storage.
