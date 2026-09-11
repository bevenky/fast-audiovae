# Complete-constraint direction probe execution

11 September 2026. The two-arm diagnostic was uploaded to the existing Runpod and launched under the qualified PyTorch runtime. No full recovery or GRAIL combination has started.

| New source | SHA256 |
|---|---|
| startup_constraint_projection.py | 88b894a8653a7989b7de9d4e5ec2a10f7c84e92e8c85397fadbaea5f21bc0042 |
| startup_anchor_direction_probe.py | 4c8e583a8d70482e062eaa4853516c0e178e8b9eca532603963b33562c00f540 |

Sixteen focused solver tests and three probe tests passed before launch. Tests covered complete linear constraints, redundant rows, normalization, independently checked optimality, same-proposal comparison, actual nonlinear acceptance, restoration and aggregate-only output. Independent source reviews found no blocker.

The existing first-zero diagnostic is pinned to SHA256 `306e4f083b831fdd877b26b12c0b9afd385cc86fda5d6cfaa25a802461ec4e43`. The original 64-update anchor aggregate and original queue configuration are authenticated by the wrapper. Both new source hashes were verified remotely before execution.

Remote root: `/dev/shm/fast-audiovae-startup-retention-audit-20260911-v1`. The new aggregate is `direction-probe-v1-aggregate.json`; its raw log stays on Runpod. Old files are not overwritten. Only the new aggregate will be retrieved.

The replay requires the exact recorded non-timing prefix through update 43 and restores the original pre-update state for both candidate directions. Every positive fraction is checked against all six calibration anchors, with twelve original inequalities. Development and ordinary reconstruction are observed after acceptance. All candidate states are restored; no new checkpoint is saved.

The diagnostic completed in 140.04 seconds, including preparation and preservation checks. All 43 non-timing scalar records matched, all 516 ordinary-source identities followed the original prefix, and every preservation check passed. The complete projection accepted a half-step versus 1/32 for the original projection, with lower same-batch ordinary objective. This is a disposable direction result, not a saved trained model. See [completed findings](direction-probe-findings.md) and [locked comparison](direction-probe-plan.md).
