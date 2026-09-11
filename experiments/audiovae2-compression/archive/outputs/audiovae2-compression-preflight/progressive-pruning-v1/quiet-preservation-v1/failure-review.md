# Quiet-preservation recovery: stopped before first update

The independent quiet-preservation trial launched after GRAIL completed, but stopped before its first completed optimizer update. No trained Q candidate or 2,000-update quality result exists. The finite controller also stopped, as designed; the training-review follow-up has been paused. No automatic restart, threshold change or code change was made.

The exact source-defined error was:

`Grad-enabled constraint differs from its no-grad pre-state`

Initial quality, full candidate state, original RNG and fresh optimizer parity all passed. The initial checkpoint was saved. There is no training journal. The completion receipt records zero completed updates and confirms preserved frozen state and protected files. Storage was not the problem: 1,617,657,856 bytes remained free in shared memory. Process, controller and configuration identities were verified, and neither the controller nor Q is still running.

The guard in `quiet_projected_update.constraint_gradients` compares a differentiable constraint's scalar value with the previously measured value at the same weights. It requires exact equality. That requirement failed before ordinary recovery could produce its first recorded update. The current evidence does not establish whether the difference is harmless arithmetic variation or a substantive mismatch in the gradient and evaluation paths. Passing initialization parity does not validate this newly exercised constraint path.

The next review should identify the magnitude and origin of that difference using an explicitly bounded diagnostic, with teacher targets, model state, input geometry and original acceptance limits held fixed. Do not silently relax the guard, change quiet thresholds or restart the run. This technical failure supplies no evidence that quiet preservation works or fails as a training method.

All failed-run artifacts remain at `/dev/shm/fast-audiovae-quiet-recovery-20260911-v1/results`, including `completed.json`, `initial-parity.json`, the step-0 development reports and checkpoint. The existing source archives, G/D final checkpoints and all earlier experiments remain preserved. Keep the pod running to retain shared-memory artifacts.

Only allowlisted error text, aggregate counters and preservation flags were returned. No audio, latents, model weights, per-recording values or identifiers were retrieved, and no new inference, tests or training were run for this failure review.
