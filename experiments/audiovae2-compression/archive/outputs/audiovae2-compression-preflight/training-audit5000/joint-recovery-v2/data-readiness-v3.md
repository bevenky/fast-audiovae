# Continuation data readiness

The final source plan preserves the exact original 27,000-source plan prefix, including its 3,000 unused reserved sources. It adds 49,500 distinct sources from already downloaded training audio. Together these supply 52,500 new crops for optimizer steps 5,625–10,000 at accumulation 12.

Source IDs, prepared-audio hashes and parent recordings are excluded against the full original plan and all reserved held-out identities. Speaker disjointness is not claimed. The same crop rules, frozen encoder and frozen teacher target-generation operations are retained.

| Fresh source interval | Crops | Scored hours | Language labels | Indic languages | Broad expressive samples |
|---|---:|---:|---:|---:|---:|
| 24000:36000 | 12,000 | 8.177 | 111 | 22 | 17.80% |
| 36000:48000 | 12,000 | 8.173 | 111 | 22 | 17.49% |
| 48000:60000 | 12,000 | 8.175 | 111 | 22 | 17.43% |
| 60000:72000 | 12,000 | 8.165 | 111 | 22 | 17.62% |
| 72000:76500 | 4,500 | 3.065 | 111 | 22 | 17.56% |

The language order is deterministically interleaved across the full extension so low-resource groups are not exhausted in early segments. Broad expressive coverage uses source-dataset labels; it does not prove that a selected crop contains a specific expression. Quiet, near-silence, startup and interior coverage are measured on each sealed shard’s exact teacher targets.

The new cache holds at most 6,000 sources. Only its regenerable `pairs.pt` files can be removed, after a durable checkpoint and receipt authenticate the complete consumed-source ledger. The cache retains manifests, source provenance, coverage, retirement receipts and checkpoint bindings. Existing caches and all source audio are preserved. Shared/exclusive file locks protect reads against retirement and marker publication.

Focused tests: 23 passed locally and 23 passed remotely. Tests include unchanged teacher-generation operation parity, source exclusion, weighted language ordering, ledger authentication, retirement safety and availability of the next 6,000-source window after checkpointing.

Producer PID: `1044621`. Initial authorized target interval: `24000:36000`.
Plan: `/tmp/fast-audiovae-continuation-plan-v3/plan.json`.
Plan identity: `f9b7ef7a9b567c350bd87435b423c2e178f110645effe98ae93a3fcf2d411e74`.
Plan file SHA256: `8054b11514b33cbd9af2f196c3d4405061bc8208ddbe9623afd2a3ada3b66bbe`.
Cache: `/dev/shm/fast-audiovae-continuation-shards-v2`.
Progress: `/dev/shm/fast-audiovae-continuation-shards-v2/producer-progress.json`.
Launch receipt: `/tmp/fast-audiovae-joint-recovery-v2/producer-24000-36000.json`.

The earlier v2 source-order plan remains preserved and is not the active plan. No training audio was downloaded to the local Mac.

## First training data wait

At 15:11 UTC, the producer was running with 1,200 targets sealed through fresh cursor 25,200 and 1,320 targets generated in 223.4 seconds, approximately 5.9 sources per second. Both the requested 24,600–24,900 shard and the following shard were complete and present. There was no producer error or cache-release wait. Only 1,200 sealed sources occupied the 6,000-source cache allowance, and shared memory had 25.72 GB free. This was ordinary waiting for an immutable 300-source shard to finish, not a producer failure or cache deadlock. Current production time was roughly 50 seconds per shard; this is an observed throughput sample, not a guaranteed run-wide rate.

The compact observation is saved in `producer-status-first-wait.json`.
