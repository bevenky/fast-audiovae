# Bounded correction probe execution

11 September 2026. The approved disposable first-zero diagnostic completed on Runpod in 218.02 seconds. It exactly reproduced the recorded non-timing scalar trajectory through update 56, then compared correction at that fixed state. All preservation checks passed. No checkpoint or automatic continuation was requested.

Nine focused CPU tests passed in 0.60 seconds after the final preservation guards were added; the parent independently confirmed 9/9 in 0.67 seconds. Independent method and source reviews cleared the signed solver and diagnostic.

| Uploaded source | SHA-256 |
|---|---|
| startup_signed_constraint_projection.py | fe3b0c8e15569079ea645b0b32539d3b0978e05f2c41e3022e303c5a062b70b5 |
| startup_anchor_correction_probe.py | 9ea35174a60169831e7b85ee37bc2f4988977b979273c22d78f1088142eb0062 |

Remote hashes were checked before launch. Frozen prior sources remain unchanged. The driver authenticates the complete64 result, original model and empty optimizer, full 768-source reference prefix, initial startup/full96/fitting observations, all 56 non-timing update records, all 90 Adam counters and exact first-zero event. Counterfactual scoring performs no optimizer update. Pre56 model, optimizer, RNG and gradient-slot identity are checked, then the original fresh state is restored on exit.

Output: `/dev/shm/fast-audiovae-startup-retention-audit-20260911-v1/correction-probe-v1-aggregate.json`. Only aggregate statistics may be downloaded. Audio, latents, weights, identifiers and raw logs stay on Runpod.

The bounded policy is in [the plan](correction-probe-plan.md). One signed solve with the doubled normal passed all six calibration starts while retaining essentially the full original proposal and improving the same-batch total objective 3.99%. This qualifies the next separate fresh 64-update recovery pilot; no GRAIL combination has started. [Findings](correction-probe-findings.md).
