# Matched comparison data

The approved 500-update comparison now has one immutable sequence for both arms. Each arm receives 16,000 windows at batch size 32, covering **9.950 hours** from **4,677 recordings**. Each scored interval occurs once per arm. Their common parent’s 11,040 used recordings remain excluded; independent older experiments do not exclude training audio.

| Allocation | Scored hours | Share |
|---|---:|---:|
| English | 2.985 | 30.00% |
| All 22 scheduled Indic languages | 2.985 | 30.00% |
| Other languages | 3.483 | 35.00% |
| Explicit vocal events | 0.498 | 5.00% |

The plan covers **110 speech languages**. Indic languages receive about eight minutes each; other-language groups receive about 2.4 minutes each, including Mandarin, Cantonese, Arabic, Latin American Spanish, Brazilian Portuguese, French, Japanese and German. This is diagnostic exposure, not evidence of qualified performance in every language. The other-language bucket includes 77.95 seconds of Japanese JNV nonverbal and JVNV mixed material; it is not 35% pure ordinary speech. These 12 files carry explicit mixed/nonverbal condition labels. Japanese FLEURS contributes a separate 65.46 seconds of ordinary speech. Version 2 changes only those 37 condition labels, preserving every scored interval, source and ordering from version 1.

The event allocation includes 15 crying/sobbing, 45 giggle, 15 shout and 14 human-whistling sources, plus laughter, screaming, whispering, breathing and Yell. Event labels describe source files; their reported durations are not precise annotations of every audible event. Yell and Shout remain separate. No scarce category was replayed and the 5% target was met.

Crops retain up to 29 preceding latent frames as unscored causal context. Scored windows contain at most 64 latent frames, including masked partial tails with at least 4,098 valid output samples. No interval wraps into the next utterance.

The metadata checks exclude 5,631 held-out source records and the fitted diagnostic sources by source, hash, parent and available real speaker/session identity. Shared FLEURS unknown-session placeholders for Spanish, Portuguese and French are explicitly disclosed. They do not prove speaker independence, and no acoustic near-duplicate audit is claimed.

All 4,677 selected audio paths exist on Runpod. The plan uses integral sample counts from the previously prepared manifests; file hashes and decoded lengths must still be checked by the target-preparation path before training. No audio was downloaded to the Mac, acquired or decoded by this planner.

The dedicated loader rechecks file digests, lineage, held-out exclusions and exact ordered windows. Six metadata tests passed, covering per-student exposure, unknown-placeholder handling, known-identity leakage, rare events, partial tails, exhaustion, mixed Japanese labels and tamper rejection.

Remote plan: `/workspace/fast-audiovae-convnext-20260909-r6/data/comparison-v2`. Plan identity: `df1a86a25ecda864b1e0843330d5956562176b3cad06174049ce3bcfb749ea7b`.

[Detailed allocation and provenance](data-plan.json)
