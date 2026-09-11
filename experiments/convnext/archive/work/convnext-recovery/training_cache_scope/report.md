# Training-cache provenance audit

This audit read the existing Runpod code, JSON receipts and sealed window plans. It imported no Torch, ran no models, regenerated no targets and changed no checkpoint or remote file.

The evidence does **not** show that every prior checkpoint was trained on incorrect latents. The original 10,000-step run used singleton teacher preparation. The later training recipes used a batch-capable path with an insufficient one-time qualification. The current step-8090 lineage was initialized afresh under that later path, so its training distribution requires a canonical-encoder evaluation. The fraction of its training examples actually affected remains unknown.

| Phase | Teacher-preparation evidence | Recorded exposure and limits |
|---|---|---|
| r1 speech warmup and original 500-hour-corpus run | Archived and deployed `corpus_training.py` / `source_training.py` use per-source caches; `SourceCorpus.get()` calls singleton `prepare_utterance_cache()`. No teacher prefetch batching exists in these versions. | Warmup: 1,000 updates, 230 training utterances, 5.0244 hours of sampled exposure. Expanded run: 10,000 updates, 371.6965 unique scored hours. These were not exposed through the later batched-encoder path; this is not a blanket guarantee against unrelated backend issues. |
| r2 corrected waveform diagnostic | First verified deployment of `batched_teacher.py` v1. Initial receipt records 31 batched utterances and one singleton, across 32 diagnostic/sentinel sources. Seven-source preliminary qualification passed. | 500 diagnostic updates; 32 training crops plus 32 sentinel crops. Actual later batches were not numerically compared with singleton references. |
| r3 objective comparison; r4 quiet calibration | r3 identity verifies reuse of the r2 parent cache. r4 reports zero teacher-inference calls and discarded diagnostic weights. | These inherit existing targets rather than independently establish target correctness. r3 arms stopped at steps 1,900 and 2,000. |
| r5 representative reconstruction | Same batch helper v1. Heldout-prefill receipt records 84 batched utterances. Training calls batch-capable prefetch. | Fresh student, 32,275 optimization windows, 20.0071 scored hours; final step 1,009 includes its preceding calibration/warmup accounting. Exact training batch/serial totals are not retained. |
| r6 balance and r7 quiet-phase comparisons | Same helper; initial training prefills record 27 and 28 batched utterances, respectively. Shared fixed heldout panel was regenerated with batch preparation and copied forward. | r6: 16,000 windows and 9.9495 hours per arm. r7: 8,000 windows and 4.9968 hours per arm. These are matched experimental exposures, not a corruption count. |
| r9 fresh recipe, steps 1–5,900 | Fresh student/discriminators/optimizers; same v1 helper, called by `recipe_v2_pilot.py`. Initial prefill records 21 batched utterances. | 188,800 committed windows; 42,426 distinct source IDs; 121.5294 scored hours. |
| r9 expressive continuation, steps 5,900–8,090 | Exact continuation; same v1 helper, called by `recipe_v2_continuation.py`. Initial prefill records 27 batched utterances. | 70,080 committed windows; 17,747 source IDs; 44.9668 scored hours. Combined optimization exposure to step 8,090: **258,880 windows, 60,073 distinct source IDs, 166.4962 scored hours**. There are 100 shared source IDs across the two segments, so source counts must not simply be added. |
| Corrected forks, step 8,090→8,490 | `run_corrected_screen.py` prepares whole sources through the same batch helper, then shares the frozen crop pools across arms. | All pools together: 17,328 distinct cached crops from 4,749 source IDs. Each generator arm consumes 12,800 windows / 8.9778 hours. Targeted-generator pool: 3,621 source IDs; regular: 3,352. Extra pools contain 2,048 discriminator warmup and 1,024 gradient-calibration windows. |

The r9 normalization-calibration plan separately contains 512 windows from 178 sources, representing 0.3419 scored hours. Its two fixed-weight passes reuse those calibration windows. This is not extra optimization exposure and is not evidence that those latent values were correct. The plan used the same batch-capable cache path.

## What the old qualification actually established

The historical helper hash is `e433f873bc725e8a1236a7e8be0782fdfb59a08bbd65745729aaca292f12a4ad`. It is pinned by the r2/r3/r6/r7 run identities and both r9 run identities, and by the corrected-screen source inventory. Archived source packages independently contain those same bytes.

`prefetch_corpus()` qualified one mixed-length sample group and stored the result under `(config, corpus.identity_sha256)`. Subsequent calls reused that process-local result. Eligible later batches ran `_batch_outputs()` and passed shape, finite-value and cache-hash validation, but **did not compare that actual batch with singleton encoder outputs**. Runtime exceptions triggered fallback; silent numerical errors did not. Cached record identities deliberately mirrored the singleton format and did not record the actual batch shape, first-call order, algorithm or per-batch numerical comparison.

Consequently, source/plan counts provide a bounded inventory of exposure to the vulnerable preparation path. They cannot reconstruct the exact count of incorrect training examples. Initial-prefill counts are only the first preparation group, not whole-run totals. A cache hit also does not tell us how that original record was generated.

## Confirmed error versus unmeasured scope

The known failing historical heldout batch contains eight utterances at padded input length 93,440. The parent audit established an encoder error affecting all eight in that reproduction. The main heldout receipt records **84 batched utterances in 11 batches**, not 84 batches. Its preliminary qualification tested five utterances padded to 321,280 samples each. This does not establish that the remaining 76 heldout utterances are incorrect, nor that the training corpus has the same failure rate.

The teacher decoder nominally generated its target from the same latent tensor that entered the cache. Thus an encoder defect can produce a changed latent/target distribution even where a cached decoder pair is internally consistent. Agreement when decoding a few cached latents is not proof that every cached target or the original waveform reconstruction is correct. Canonical full-source evaluation and the separate decoder/backend audit are still needed before making that stronger claim.

## Evidence files

- `remote-inventory.json`: source hashes and pruned existing receipts across r1–r9. List lengths are explicitly labeled where content is abbreviated.
- `remote-exposure.json`: exact source counts and hours from verified sealed r9 plans, plus run-identity hashes and the corrected-screen pools.
- `early-prefills.json`: complete r2 and r5 batch-membership receipts and the deduplicated r9 source union.
- `remote_inventory.py` and `remote_exposure.py`: standard-library-only read commands used for this audit.

No claim here treats the current newly patched helper as evidence of how historical caches were produced. Historical checkpoints and caches remain preserved; the canonical panel is a separately versioned evaluation domain.
