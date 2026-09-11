# Fresh training source handoff

The sealed v2 ledger has 27,000 distinct sources. Only its first 12,000 are approved for the step1000 to step5000 continuation. The remaining 15,000 are a conditional reserve and the producer will not generate them.

Original fitting, calibration, development and known reserved source IDs, audio hashes and parent recording IDs are excluded. No speaker disjointness is claimed. The first300 previously unused cached sources passed the real CPU loader check; seven focused loader tests also pass. All source audio stays on Runpod.

## Live locations

- Plan: `/workspace/fast-audiovae-compression-20260910-v1/fresh-source-plan-v2/plan.json`
- Plan identity: `3c65151fd2b37900257e179fee62218c3d33055c94d47dbb7c213593777e7c7d`
- Shards: `/dev/shm/fast-audiovae-compression-fresh-shards-v2`
- Producer PID: `1018105`
- Producer log: `/workspace/fast-audiovae-compression-20260910-v1/fresh-pair-producer-v2.log`
- Launch receipt: `/workspace/fast-audiovae-compression-20260910-v1/fresh-pair-producer-v2-pid.json`
- Progress receipt: shard root plus `producer-progress.json`
- Final receipt when finished: shard root plus `producer-complete.json`

Each300-source directory is published by writing `pairs.pt`, `source-provenance.json`, then `receipt.json` last and atomically. The loader only consumes completed receipts and checks file hashes, identities, ordering, tensor geometry, dtype and finiteness. The trainer waits without moving its source cursor if a shard is not yet sealed.

## Target generation

For each new source, the producer authenticates original audio bytes and prepared mono16kHz samples, performs one full-source frozen FP32 encoder forward with cuDNN disabled, and decodes the entire latent sequence at48kHz. It repeats the decoder forward and requires bitwise equality. Each previously unseen source length receives three decoder warmups. Only afterward does it select the sealed crop and30-frame history, with the real tail count preserved. No optimizer or student update runs in this process.

The initial measured producer rate was about5.6sources/s. This is slower than the resumed trainer can consume data, so target preparation currently limits wall time. The strict repeat checks are deliberate; no throughput or recipe changes were made during this continuation.

## Coverage and limitations

The approved12,000 sources contain all22 scheduled Indic languages and the requested international speech groups. They comprise31.59hours of source recordings and8.16hours of scored crops. Broad expressive datasets contribute20.16% of scored crop duration, including280 FSD vocal sources and two human-whistling sources. These are source-level labels, not timestamped proof that every selected crop contains the labeled event. The remaining reserve contains264 FSD vocal and seven whistle sources.

The v1 selection is preserved but unused. It was replaced before training because language-weighted selection missed the rare whistling sources. The v2 plan identity and ordered source list have not changed after launch.
