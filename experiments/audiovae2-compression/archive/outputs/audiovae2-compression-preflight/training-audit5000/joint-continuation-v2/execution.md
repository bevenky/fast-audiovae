# Joint continuation toward 10,000 updates

The user authorized the recommended continuation, retaining the current model and checking progress in bounded segments. The retained starting checkpoint is step 5,625, SHA `0da2f6e98a29b025b41df84dbb674e1a35fbea2629f698ef33fe62f687c18b1b`.

Stages 2, 3 and 4 remain jointly trainable. The encoder, teacher, prefix and suffix remain frozen. AdamW stays at 0.00003 with the existing coefficients, accumulation of 12 unique singleton sources, saved moments and RNG. No reconstruction objective, architecture or acceptance threshold changed.

The first segment targets step 6,625, with a review at 6,125. Later segments require a completed review bound to the preceding checkpoint, source cursor, source plan and runner hash. Each advances at most 1,000 steps. The final segment is 375 steps, from 9,625 to 10,000. History persists across segment boundaries. A failing review pauses training for diagnosis and never freezes layers automatically.

The original 96-recording panel is retained. Added observations separate first-20-ms student startup, 20–40-ms teacher transients on zero input, sustained source silence, interior near-silence and other quiet audio. They share the existing evaluation pass. Both residual RMS and excess over the existing output-level limit are recorded, so a binary pass-rate plateau cannot conceal continuous progress. Original quality values remain visible.

The continuation needs 52,500 unique source crops. It consumes the unused 3,000-source reserve first, followed by 49,500 new selections from existing Runpod audio. All original plan entries, source/hash/parent exclusions and crop geometry are retained. The final ordering distributes all 22 Indic languages across all five segments. The producer measures actual quiet and near-silence sample coverage; language or expressive labels are not treated as event-duration measurements.

The new cache is bounded at 6,000 source crops. Only its regenerable, consumed tensor files can be removed, after a saved checkpoint and its source ledger have been verified. Receipts, source records, original caches, raw audio and checkpoints remain. Shared and exclusive locks prevent readers from observing retirement before its verified cursor is published.

## Verification before launch

The continuation runner and gates passed 46 focused CPU tests on the Runpod software environment. Tests cover the unchanged optimizer update, exact restoration, source uniqueness, checkpoint/provider integration, review history, the final 375-step segment and quiet-region accounting. The data path has separate focused planner, cache and retirement tests. Independent review cleared the integrated methods and resolved the pre-launch ledger-schema and cache-capacity issues.

The actual restored model must reproduce the saved step-5,625 quality report before any training update. Every training crop retains a live frozen-teacher comparison with the authenticated cache. New live results and launch receipts will be saved beside this note.

## Initialization clarification

All three stages were jointly trained from the first optimizer update of this pruned-model experiment. The authenticated step-1,000 checkpoint contains 90 optimizer states, 30 per stage, including conditioning parameters. All have step counter 1,000 and nonzero first and second moments. The current model-construction source hash matches the original experiment record. [Early checkpoint evidence](../joint-recovery-v1/early-stage-training-evidence.json).

The student started from surviving teacher weights, not random weights. Pruning changes their combined function because removed channels no longer contribute. Earlier controlled restoration experiments demonstrated that this can alter silence and startup behavior. They do not prove that inherited initialization is intrinsically harmful or that random initialization would fix the current residual. Reconstruction after pruning is established practice, while published comparisons also show inherited weights are not universally superior to training a pruned architecture from random initialization. [Channel pruning](https://arxiv.org/abs/1707.06168), [initialization comparison](https://arxiv.org/abs/1810.05270).

No random-initialization experiment was launched. The present decision is to retain the recovering checkpoint and test whether continued learning improves the separately measured gaps.

The continuous dashboard uses the new campaign's shared raw log directory. Historical dashboard files remain preserved. The campaign does not commit, push or promote the model automatically.

## Live launch verification

On September 10 at 15:12 UTC, the continuation was running at step 5,767 after 142 updates. Restored group weights, optimizer state and RNG matched exactly. Baseline quality reproduced the saved report within its fixed tolerances. All 1,704 teacher/cache comparisons were bitwise equal, and all 1,704 source identifiers were unique. No failure receipt existed. These are launch checks, not a new claim of improved quality; the first full review remains step 6,125.

The TensorBoard server now serves this campaign with 14 labeled metric runs plus Details. It retains the previous logs and adds separate startup, sustained-source-silence and interior-near-silence pass rates. The five-minute review follow-up is active. It may advance a completed segment only after a favorable review and valid preservation, checkpoint and source checks; otherwise it pauses.

The producer runs alongside training using the fixed v3 plan. Brief waits for sealed 300-source shards were ordinary preparation waits, with no producer error or cache deadlock. At the observed throughput, preparation can temporarily determine training pace. No training recipe change was made in response. [Live verification](launch-verification.json).
