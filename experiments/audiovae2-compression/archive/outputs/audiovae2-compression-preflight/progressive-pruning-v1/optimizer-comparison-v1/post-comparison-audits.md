# Audits after the optimizer comparison

The user added protection speed to the pending work on 11 September 2026. Finish the current optimizer qualifications and independent recovery comparisons with the frozen recipe. Do not change protection, loss weights or runtime during them. The ordinary-quiet regression investigation remains required and must not be displaced by this speed work.

## Track protection now using existing logs

The first seven completed 64-update trials spend75.1–79.8% of core update time in protection. Total protected update cost is4.01–4.95 times the ordinary update region. These are paired regions within the same training steps, excluding initial setup, validation and other outer work; they are not whole-job or decoder-inference speedups.

The original AdamW control records0.5885 seconds per ordinary update plus2.1180 seconds of protection, totaling2.7065 seconds. The high-rate Muon trial records0.6414 plus2.5309 seconds, totaling3.1723 seconds. Protection remains the main training-time cost regardless of the matrix optimizer.

[Current aggregate timing evidence](protection-timing-20260911T134431-aggregate.json). TensorBoard already records `update/q_ordinary_update_seconds` and `update/q_auxiliary_seconds` each step. No new in-run instrumentation is needed to track those totals. [Timing charts](https://34d6pb4ub5ldrz-8888.proxy.runpod.net/#scalars&tagFilter=%5Eupdate%2Fq_%28ordinary_update%7Cauxiliary%29_seconds%24).

Counters also record canonical score forwards, constraint-gradient forwards, normal solves/accepts, fallbacks and actual movement. They identify work volume, not which component dominates elapsed protection time. Separate QP, Gram, reverse-pass and state-copy timings are currently missing.

## Audit 1: quiet-window regression

After every optimizer arm has a completed result or explicit failure disposition, compare the same saved windows at matched steps, including group-only snapshots at500/1500. Keep fixed thresholds, masks and denominators. Count pass-to-fail and fail-to-pass transitions; separate amplitude-only, residual-only and combined failures. Inspect all seven overlapping cohorts, and keep six protected calibration starts distinct from13 held-out starts and ordinary quiet. Do not optimize against development data or label a lower mean RMS as fewer failures.

This begins with saved evaluation records, without another training run. If the records only classify a symptom, say so. Any subsequent controlled replay must preserve the original checkpoint, optimizer, RNG and data and return only aggregates. Six protected startup cases do not imply that ordinary quiet audio is protected.

## Audit 2: protection latency

Use a separate disposable replay after the comparisons, preserving the exact state and startup policy. Separate setup, ordinary update, constraint forward/reverse computation, Gram construction, solver, nonlinear rescoring, state copies, validation and device synchronization. Use synchronized diagnostic timing and an uninstrumented control to expose measurement overhead. Preserve all original artifacts and do not advance or resume a comparison model.

Prioritize these mechanisms:

1. **Small-QP enumeration.** `startup_constraint_projection.py::solve_small_qp` evaluates all4,096 subsets even after certified candidates are found. Test candidate ordering or an early exit only when the same full12-constraint primal/KKT certificate establishes the solution. Preserve rank-deficiency handling, deterministic tie policy and the final waveform gate. Do not assume any first feasible subset is optimal.
2. **Gram/dot and transaction work.** `quiet_projected_update.py::_dot` repeatedly scans90 tensors, casts toFP64 and converts scalar results to Python. Profile allocation, reductions, host/device synchronization and state copies separately. Test cached immutable geometry and consolidated calculations only with qualified numerical equivalence; do not cache changing Jacobians.
3. **Repeated anchor passes.** One complete startup Jacobian uses six forwards and twelve reverse traversals; rescoring and normal corrections add work. Investigate shared computation or verified duplicate elimination only with original waveform, gradient and accepted-update parity. Previously observed batching sensitivity must not be bypassed by simply batching the anchors.

Every candidate must retain all12 constraints, all six calibration anchors, existing thresholds, per-update checking and nonlinear acceptance. Validate the actual accepted displacement and quality checks, not just the proposed gradient. Report ordinary compute and full protected-step wall time, including overhead. Meaningful speed gains remain unproven until measured.

No weaker thresholds, less frequent checking, fewer anchors, architecture changes, new inference benchmarks, commits or model promotion are part of these audits. After the audits, recommend the smallest well-supported changes. Preserve both audit results and all prior optimizer comparisons.
