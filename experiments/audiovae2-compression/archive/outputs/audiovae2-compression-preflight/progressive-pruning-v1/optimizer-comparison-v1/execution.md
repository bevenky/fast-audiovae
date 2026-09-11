# Optimizer comparison execution

The user approved the remaining comparison plan on 11 September 2026. Qualification was dispatched at 13:13:12 UTC. Controller PID 1113091, start tick 4023890079. Its first child is AdamW at pointwise learning rate 0.00003. A completed optimizer comparison result is not yet available.

Remote root: `/dev/shm/fast-audiovae-optimizer-comparison-20260911-v1`.

- [Declared plan](plan.md), [machine configuration](plan.json), [source formulas](optimizer-sources.md).
- [Source freeze](source-freeze.json), [installation receipt](source-install.json), [dispatch receipt](qualification-dispatch.json), [local test results](test-results.json).
- Runner hash: `6ac8954ac0bc5f4cd6a18dd169e37190c3624d731a8809f324b608095e6f6147`.
- Controller hash: `d5cc30b0cee8124b7e4de292bd339f35468ac3845a8f616f09023829a79b64b8`.
- Plan hash: `c219369ccc09efe9e7b0d7843f778a4d0e38f7d1d6dcac3efa6669e58bf195a7`.

The uploaded archive is 35,307 bytes and contains only new source, tests and configurations. It reuses existing immutable dependencies, weights and data. Torch 2.14.0+cu126 and cuDNN 92501 are unchanged. No package installation, audio download, commit or push was performed.

Qualification has 12 separate fresh 64-update trials. The exact AdamW control runs first and must reproduce all non-timing scalars of the previous corrected64 run. Select each arm's rate on the separate training-calibration probe only. Then use the sealed decision to dispatch four independent fresh 2,000-update comparisons. A method with no eligible short trial is reported as such and cannot enter recovery. No checkpoint is used to initialize another trial.

The controller serializes experiments and authenticates configuration, optimizer activity, data order, source hashes and startup checks. Unknown failures stop for review, rather than being relabeled numerical failures. Storage is checked before each arm and snapshot. Existing experiment artifacts are never deleted or overwritten.

The current hardware has about 1.385 GB free shared memory at launch. The measured native group occupies 26,272,768 bytes. The controller requires 434,927,616 free bytes before an AdamW/Muon/NorMuon recovery and 567,048,192 before Shampoo. It saves group-only snapshots at 500 and 1,500, and full states at 0, 1,000 and 2,000. These additional snapshots support the deferred quiet-window audit.

## Reading progress safely

Use `work/convnext-preflight/runpod.py` and verify each PID against its `/proc` start tick. `qualification-progress.json` and `recovery-progress.json` contain aggregate progress. Child launches are in `launches/`; each output directory has aggregate `completed.json` and `review-step*.json`.

Raw `train.jsonl`, `launch.json` and `development-step*.json` contain source identities or individual-window values. Parse those on Runpod and return only aggregate statistics and hashes. Never download audio, latents, model values, checkpoints or raw private logs.

The TensorBoard URL remains https://34d6pb4ub5ldrz-8888.proxy.runpod.net/. Switch its authenticated server to the new comparison events only after an actual update. Old events remain on disk. Report the change in `tensorboard-launch.json` and here.

## After qualification

The controller writes `qualification.json` plus its hash after all twelve candidate dispositions. Review its shared fresh initializer/RNG/probe/source identities, control parity, all preservation flags and genuinely nonzero updates. The selected learning rates must depend only on `training_probe_after.total`. No development-based filtering or selection is allowed.

The already-authorized recovery dispatch is:

```sh
/usr/bin/python3 /dev/shm/fast-audiovae-optimizer-comparison-20260911-v1/launch_optimizer_comparison.py --plan /dev/shm/fast-audiovae-optimizer-comparison-20260911-v1/plan.json --phase recovery
```

This dispatch is allowed once qualification has passed. It is not a new architecture or training recipe. The child verifies the sealed decision again. Never rerun a completed controller phase or overwrite an existing destination.

Once all four comparisons have a disposition, audit the ordinary quiet-window regression using the saved matched evaluations. Keep calibration startups, held-out startups, broader quiet and near silence separate. Correlation is not perceptual accuracy. There is no model promotion, new width cut or inference benchmark in this plan.

## First live verification

At13:15 UTC the first AdamW trial had39/64 completed updates. Every non-timing recorded scalar matched the previous corrected64 reference, all six protected calibration starts passed, and every update moved parameters.

TensorBoard switched at13:16:23 UTC to this experiment's event directory: PID1113419, start tick4023909219. Old events were preserved. The existing review heartbeat is active and includes the already-approved transition from qualification to fresh2k comparisons.

At13:17 UTC the exact AdamW control completed all64 updates and qualified: non-timing scalar parity passed,768 distinct ordinary sources,64 nonzero updates, genuine optimizer counters and all preservation checks passed. Its completion hash is `29abb5ee06c74551f93775ed7808f6d48dfc59ab4a91b8f9d073b9ef3202d571`. The controller then launched `adamw-low`, PID1113523/start tick4023916392. TensorBoard served64 loss points for the completed control. [Qualified control receipt](adamw-control-qualified.json).

## Added follow-up: protection latency

The user requested this after the comparisons. Existing timing totals show4.01–4.95 times ordinary update cost across the first7 completed qualifications. Protection accounts for75.1–79.8% of core step time. The same logs and TensorBoard already track both regions. Add detailed protection profiling after the experiments, retaining every physical check and preserving all results. The quiet-window regression audit remains required. [Both follow-up audits](post-comparison-audits.md).

## Qualification audited; fresh recovery dispatched

11 September 2026, 14:15:33 UTC: all 12 qualifications completed and passed the independent CPU audit. The audit rehashed 72 protected files, reproduced all candidate dispositions and training-probe-only selections, and verified all 768 completed update certificates. Every trial made 64 nonzero updates on the same 768 ordinary sources. The exact AdamW control passed recorded non-timing parity. Qualification SHA: `25ac5da9f67015086962b9249a87a74dde74f1df02837218af28ebbc1756b809`. [Audit receipt](qualification-completion-audit-aggregate.json).

Selected pointwise rates are AdamW `6e-5`, Muon `1.5e-4`, NorMuon `1.5e-4` and Shampoo `6e-5`; the other 81 parameters retain AdamW at `3e-5`. Selection used the fixed non-anchor training-calibration probe, with no development selection. Each selected endpoint passes 6 / 6 calibration starts and 12 / 13 held-out starts. Broad quiet remains unresolved, and there is no optimizer winner or model promotion. [Selected quality and timing table](update-20260911T1419.md).

The already-approved fresh recovery phase was dispatched at **14:18:27 UTC**. Controller PID `1119736`, start tick `4024281579`; first AdamW child PID `1119737`, start tick `4024281582`. At **14:19:50 UTC**, the authenticated training journal contains **13 / 2,000 actual updates and 156 ordinary sources**, with every completed startup check passing. Fresh initializer/RNG provenance and the step-zero checkpoint hash pass. The periodic progress file had an earlier 10-update observation at 14:19:42. The newer journal is the source for the 13-update status. [Recovery status receipt](recovery-status-20260911T141950-aggregate.json).

Muon, NorMuon and Shampoo are queued serially, each from the same fresh initializer and an empty optimizer, with no qualification checkpoint resumed. The sealed plan hash remains `c219369ccc09efe9e7b0d7843f778a4d0e38f7d1d6dcac3efa6669e58bf195a7`. Source, configuration, older results and checkpoints remain preserved. The dedicated quiet-regression and protection-speed audits follow completion of these comparisons.

Latest observation, **14:21:42 UTC**: the same authenticated AdamW child reached **54 / 2,000 updates**. TensorBoard independently serves 54 `loss/total` points through step 54. Muon, NorMuon and Shampoo remain queued serially. [Step and TensorBoard receipt](tensorboard-recovery-20260911T1421.json).

## AdamW step 1,500 milestone; recovery still running

11 September 2026, 15:33:06 UTC: the authenticated journal records 1,615 / 2,000 actual AdamW updates on 19,380 distinct ordinary sources. All 1,615 updates have nonzero parameter movement and passing protection certificates. Controller and child are active; plan/configuration/source verification and saved checkpoint hashes through 1,500 pass.

The step-1,500 evaluation reaches 13 / 13 held-out startup passes alongside 6 / 6 protected calibration starts, 184 / 184 near-silence windows, 50 / 50 interior near-silence and 164 / 164 sustained source-zero windows. Correlation is 0.9930394554, waveform MAE 0.0023996012, mel error 0.1472411 and broad quiet RMS 80.526 micro full-scale, with no overshoot. Broad quiet passes only 1,144 / 2,544, while the 20–40 ms transient cohort remains 0 / 10. These overlapping cohorts are not additive, and this running milestone does not establish that silence is solved or startup passing will persist.

Muon, NorMuon and Shampoo remain queued serially. Continue the authorized bounded AdamW arm unchanged; no winner, promotion or further cut. Quiet-regression diagnosis and protection-speed profiling remain pending after the comparisons. [Milestone report](update-20260911T1533.md), [authenticated aggregate](monitor-20260911T153306-aggregate.json).

## AdamW recovery completed; Muon running

11 September 2026, 15:51:25 UTC: AdamW completed 2,000 nonzero updates on 24,000 distinct ordinary sources. The CPU completion audit passed all 40 checks, 62 immutable-file hashes, all five snapshot receipts and the native optimizer/RNG checks. Full snapshots at 1,000 and 2,000 contain 90 finite Adam states with the correct counters; step zero has an empty optimizer. The final checkpoint SHA is `5563cc49a11a9de84510f1a84cd319b1675cb3b5e7cd114f62536a5712dcccc0`.

From step 1,500 to 2,000, correlation declined from 0.99303946 to 0.99166654 and waveform MAE rose from 0.00239960 to 0.00272806 (+13.69%). Mel and group MSE improved slightly. Broad quiet passes declined from 1,144 to 1,097 / 2,544 and residual RMS rose from 80.526 to 85.226 micro full-scale. Protected calibration starts remain 6 / 6; held-out starts changed from 13 to 12 / 13 and near silence from 184 to 179 / 184. All five final near failures are amplitude-only. The 20–40 ms transient cohort improved from 0 to 10 / 10. No overshoot occurred. The cohorts overlap and these mixed changes do not establish an optimizer winner or recipe failure.

Recorded elapsed time was 91.02 minutes, with mean update time 2.673 seconds: 0.588 ordinary and 2.081 protection. Muon has started from the same fresh initializer, PID `1131527` / start tick `4024831693`. Its authenticated journal has 12 updates on 144 ordinary sources, all nonzero with passing protection certificates, and its step-zero snapshot hash matches. NorMuon and Shampoo remain queued serially. All prior results remain preserved; no model promotion or new cut. Quiet-regression diagnosis and protection-speed profiling remain deferred until all four arms have a disposition. [Completed result](update-20260911T1551.md), [CPU audit](recovery-adamw-completion-audit-aggregate.json), [Muon status](monitor-20260911T155125-aggregate.json).

## User-directed pause; shorter campaign proposed

11 September 2026, 16:04:03 UTC: the comparison is paused at the user's direction. The authenticated receipt records both controller PID `1119736` / start tick `4024281579` and Muon child PID `1131527` / start tick `4024831693` in stopped state `T`. Muon has 253 logged completed updates and reviews at 0 and 250. Its model and optimizer remain resident in process memory. Only the step-zero disk checkpoint exists; no new checkpoint was created for this pause, and the resident state must not be described as a saved step-253 checkpoint.

AdamW remains completed and audited. Muon is paused; NorMuon and Shampoo are deferred, not discarded. All prior artifacts are preserved. The follow-up automation is `PAUSED` and automatic resume is disabled.

Quiet-regression diagnosis and protection profiling now take priority without waiting for the deferred optimizer arms. The proposed shorter campaign retains all twelve physical constraints, screens `256 / 256` then `256 / 128` widths with all nine residual units, and requires a measured one-thread streaming CPU gain before substantial recovery. Its proposed budget is 64-update feasibility, review at 250 and a 500-update cap with one fixed AdamW recipe. This is a proposal; no new experiment, next cut or final recipe has started or been approved by this entry. [Pause receipt](user-pause-20260911T1604-aggregate.json), [proposed campaign](../next-campaign-plan.md).
