# Recover the first cut from update 1,000 to 2,000

The user approved another 1,000 recovery updates at the existing 384/256 widths after the first-cut review. This does not prune another boundary. Preserve the existing architecture, original frozen teacher and encoder, three losses, learning rate and all 90 joint stage 2 through 4 parameter tensors.

Resume parent checkpoint `4272198a8d76168564651c49960ef370690859981a5591ba0de39d42053eef56` from `/tmp/fast-audiovae-progressive-pruning-v1/cut1-384-256/checkpoint-step1000.pt`. Restore its weights, all AdamW moments and step counters, RNG and source ledger. Unlike a width cut, this continuation must not reset the optimizer.

Consume progressive source positions 12,000 through 23,999, corresponding to fresh-plan positions 9,000 through 20,999 after the original 3,000-source fitting cache. The completed student has already used the first 12,000 progressive positions. No source may repeat within this student. Extra recovery reduces the sources available for later pruning cuts, so the current 30,000-source stream must be extended before it is exhausted.

Use the same 96-recording held-out panel and the existing boundary panel. Verify that restored update 1,000 reproduces the saved report before new updates. Evaluate at 1,500 and 2,000, saving checkpoints and all seven quiet-region measurements. Keep actual waveform and mel errors, active waveform cosine, active signal RMS relative to the teacher, whistle RMS, peaks and quiet errors visible. A correlation milestone alone does not establish recovery.

Create a new isolated TensorBoard event directory. Replay saved scalar history once for display, preserving the actual first-cut step-zero error baseline. Render the training-progress curve consistently against the new target of 2,000 across its history. Replaying saved metrics is not another training pass or source reuse. Append new losses each update and new validation at 1,500 and 2,000. Keep every old event file unchanged.

Expected remote recovery root: `/tmp/fast-audiovae-progressive-pruning-v1/recovery-1000-2000`. Additive runner and monitor source live under `code`; model dependencies are imported from the existing first-cut code and original compression code directories. The trained state lives under `segment-1000-2000`, separate from the parent checkpoint. The existing port 8888 dashboard switches only after a real resumed update is observed.

Stop after optimizer update 2,000 and review the same-width recovery. Do not begin another cut, alter losses, add an output gain, promote the model or commit changes automatically.

Status: completed at update 2,000 and stopped awaiting review. The segment took 667.06 seconds, including its scheduled reviews. It consumed 12,000 new distinct sources, bringing this student's total to 24,000. All 12,000 original-teacher cache comparisons passed. No optimizer reset, further pruning, benchmark or promotion occurred.

Final checkpoint: `/tmp/fast-audiovae-progressive-pruning-v1/recovery-1000-2000/segment-1000-2000/checkpoint-step2000.pt`. SHA256: `a7b6c5ee850c80061c073dd42ab7a019ce96d55b0b9691b77c9721ea0b14602f`. All 90 AdamW states reached update 2,000 with unchanged settings. Weights, moments and losses were finite. Frozen-state and protected-file checks passed. The independent integrity review remains on Runpod at `integrity-review-2000.json`.

All 32 focused tests passed locally and on Runpod. Source readiness verified all 40 required sealed shards (9.647 GB), including actual tensor-cache hashes, and disjoint source IDs, audio hashes and parent-recording IDs. The restored group parameters, all optimizer state and RNG matched exactly before and after guarded starting validation. The entire saved step-1,000 report passed the existing numerical comparison tolerance (absolute 1e-6, relative 1e-5), with exact counts and structure. No optimizer reset or width change occurred.

Frozen runner SHA256: `d3a7be85429e54f6dc8199174238847d9eb5acef8dbe42ebd44feaf2f511906c`. Monitor SHA256: `9e57cc0854c326e657fae92dc398cc2cf8a1fd2e57be7efac5b18ad3278c362a`.

Reviews at 1,500 and 2,000 are complete. The follow-up monitor is paused after the final assessment. Only this same-width segment has been launched. See `review-step1500.md` and `review-step2000.md` for the compact assessments; raw reports remain on Runpod. Final browser verification shows all 15 colored overview series and the full step-zero through 2,000 history with outlier clipping disabled.

TensorBoard serves this recovery's isolated event directory on the existing URL. Its API exposes 15 overview metric runs plus Details. Browser verification confirmed the preserved step-zero, 500 and 1,000 points, new progress beyond 1,000, and the active and whistle RMS curves. Outlier clipping remains disabled and the chart is fitted to the full available history. The existing browser tab is retained for monitoring.

The remote `launch-evidence.tgz` preserves process, dashboard, CPU-test, restore-check, starting-quality and history-replay receipts. Automatic approval review rejected downloading this archive to the Mac because explicit authorization for the payload and destination was not recorded. The archive remains on Runpod; no alternative transfer was attempted. User approval for the optional local copy is pending. This does not block training or monitoring.
