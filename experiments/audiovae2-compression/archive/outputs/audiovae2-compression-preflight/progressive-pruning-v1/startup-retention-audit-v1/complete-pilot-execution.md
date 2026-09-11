# Complete-constraint pilot execution

11 September 2026. Completed the approved fresh 64-update pilot on the existing Runpod in 235.42 seconds. All preservation checks passed. No checkpoint or automatic continuation was requested. Only aggregate results were exported.

The final focused test suite passed 7/7 in 1.64 seconds. Independent method review found no blocker. Two additive source files were uploaded and their complete SHA-256 hashes checked before execution:

| Source | SHA-256 |
|---|---|
| startup_complete_anchor_update.py | 1f4880c649868e1a9861595c4ab60d4df668cbdbfb4a593a9b81cb51c5e4208d |
| startup_complete_anchor_pilot.py | 6a82dffeaf61f8d2d8e0381868e8f67ff4b8cb06f311ad85b61667609d734307 |
| Existing complete-constraint solver | 88b894a8653a7989b7de9d4e5ec2a10f7c84e92e8c85397fadbaea5f21bc0042 |

The wrapper authenticates and invokes the immutable original pilot driver, using its existing environment and exact fresh-C construction. It changes the linear constraint set from two maxima to twelve individual inequalities and uses the previously tested extended fraction grid. The original objective, Adam recipe, six calibration anchors, 768 distinct ordinary sources, 96-source development review and final restoration stay fixed. Development metrics do not select the update.

The comparison is against both completed retention pilots and the ordinary control. Its result must establish nonzero sustained movement as well as startup preservation. A pass on six calibration anchors alone does not qualify the method.

Remote result: `/dev/shm/fast-audiovae-startup-retention-audit-20260911-v1/complete-anchor-pilot-v1-aggregate.json`. Raw logs, data and model tensors stay on Runpod. The GRAIL combination remains pending this qualification; no model architecture change has been made.

The complete policy improves reconstruction and reduces zero updates, but late movement remains restricted. Preserve this result; the next bounded work is its first-zero correction diagnostic. [Findings](complete-pilot-findings.md).
