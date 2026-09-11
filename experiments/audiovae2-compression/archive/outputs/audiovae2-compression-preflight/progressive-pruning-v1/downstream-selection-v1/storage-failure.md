# First recovery: completed measurements, missing final checkpoint

The first downstream-selected recovery executed all2,000updates/24,000unique sources and all six planned development reviews. The original teacher/frozen-stage checks and protected-file checks passed. It is nevertheless a failed run because the final model/Adam checkpoint was not saved. Its original `completed.json` remains marked `failed`; it is not relabeled as success.

The final checkpoint guard raised `Insufficient disk headroom for the bounded group/optimizer checkpoint`. Only checkpoints0 and1000 exist. The2000 development report exists, but the process then exited and its final in-memory weights are unavailable.

The cause was operational: the qualification tests left152,312,276bytes of synthetic pytest fixtures on `/tmp`, consuming headroom previously reserved for the training checkpoint. At failure only70,561,792bytes remained. There was no neural arithmetic, teacher, source-ledger or quality-threshold failure. The first queue stopped correctly before launching GRAIL.

The root execution process caused this storage conflict. No prior experiment or test evidence was deleted to hide it.

A new independent repeat starts from the same fresh original-teacher factory and sealed downstream initializer, exact original RNG, empty Adam and same first24,000source sequence. It does not resume checkpoint1000. Model code, calibration, objectives and quality gates remain unchanged. New outputs, checkpoints, TensorBoard and follow-on outputs use Runpod shared memory, with over2GBfree measured against a1GBbudget for all three trials. Qualification is already complete; no more temporary test fixtures are being generated there. Keep the pod running to retain these artifacts.

New recovery root: `/dev/shm/fast-audiovae-downstream-recovery-20260911-v2`. The original failed recovery and failed queue remain immutable. Previously measured milestone scores remain a repeatability reference; they are not a deployable2000 checkpoint.

First-run2000 measurements: MAE0.00255561516, mel0.141937599, active cosine0.991852518, quiet RMS88.6941855microFS,1,158/2,544quiet windows passing, no output overshoot. These values cannot qualify a saved model because its endpoint checkpoint is missing.
