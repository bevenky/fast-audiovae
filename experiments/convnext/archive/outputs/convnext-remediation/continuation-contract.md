# Data continuation contract

The contract below has now been implemented and validated. A separate continuation runner imports the verified step 5,900 state and a sealed remaining-data plan; the original runner still accepts only an exact resume of its own immutable plan. The old unknown-language classifier is not reused for the explicitly labeled emotional nonverbal supplement.

The versioned continuation checks:

1. Stop the parent at a committed boundary and acquire its run lock. Checkpoint, global step, sampler cursor, metrics and exposure journal must agree, with no in-flight update.
2. Restore the student, discriminators, all optimizer states, gradient-balancer EMA and RNG state. Preserve frozen normalization statistics and their original calibration provenance. Do not repeat warmup or calibration.
3. Build an interval ledger from the actual committed fixed-window prefix, retaining all existing/new evaluation reservations. Reject source/hash/parent overlap with consumed intervals and calibration windows.
4. Seal a new data segment with its own manifest, ordered windows, teacher/source hashes and separate segment cursor, bound to the exact parent checkpoint and journal hashes. Preserve the original journal.
5. Consume only canonical v2 supplemental metadata. Keep generic emotional nonverbal material distinct from named actions, and allocate its share explicitly.
6. Use a new SourceCorpus cache namespace or a separately validated migration. The current cache identity binds the entire old source manifest.
7. Keep the remaining update budget explicit. Any extension beyond the original 10,000-step budget requires a declared budget transition, not an undocumented resume identity change.
8. Validate identical next-update behavior on unchanged input, state preservation, continued resume, and rejection of stale parents, uncommitted exposure, identity changes and repeated intervals.

This changes orchestration and data scheduling, not the decoder or losses. It is not implemented by silently rewriting a checkpoint identity. The original process had no runtime pause request. The implemented coordinator used a verified process descriptor, suspension followed by complete checkpoint/journal inspection, and default-termination checks to stop it at a committed boundary. The new runner supports a pause request between complete updates.

The step-3,000 snapshot is for evaluation. It must not be used to rewind a parent that has subsequently consumed more audio.

The first two continuation updates saved at global step 5,902. Exact resumed training is bounded at 10,240. See [execution report](execution.md) and [handoff receipt](committed-handoff.json).
