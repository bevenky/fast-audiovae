# Quiet phase comparison data

The next matched trial has **8,000 windows**, **2,351 source files**, and **4.997 scored hours per arm**, supporting 250 updates at batch size 32. The parent is the selected r6 control checkpoint. The ledger excludes all **15,717** source files used by this student in r5 and r6, with shared A/B data counted once. Older independent-model exposure does not exclude training data.

| Allocation | Scored hours | Share |
|---|---:|---:|
| English | 1.499 | 29.991% |
| All 22 Indic languages | 1.499 | 29.997% |
| Other languages | 1.749 | 35.000% |
| Explicit events | 0.250 | 5.012% |

There are 110 speech languages in this diagnostic sequence. The other-language allocation includes 53.77 seconds of explicitly labeled Japanese JNV nonverbal/JVNV mixed material, so it is not entirely ordinary speech.

**Rare event recordings already consumed by this student are not replayed.** The new event allocation contains 74 laughter recordings, 33 screaming recordings and 89 German whisper recordings. There are no remaining eligible crying, giggle, shout, human-whistle, Yell, chuckle, new Whispering or Breathing recordings under the conservative whole-file exclusion policy. The overall 5% event target is met, but that does not mean every event class is represented.

All selected audio paths exist on Runpod. Actual source hash and decoded-length checks remain required before frozen-teacher target preparation. Masked partial tails retain at least 4,098 output samples and up to 29 preceding latent frames as unscored causal context. Source/hash/parent and known speaker/session reservations remain enforced; shared FLEURS unknown-session placeholders are disclosed without claiming unknown-speaker independence.

The driver verified the final 500-step r6 exposure journal against the common ordered plan, the source checkpoint, the extracted selected-parent checkpoint and its selection record. The published plan successfully roundtripped through the checksum and exclusion-verifying loader. Two focused lineage tests pass, including rejection of current-student repeats and exhaustion of scarce events.

Remote plan: `/workspace/fast-audiovae-convnext-20260909-r7/data/quiet-phase-v1`.

Identity: `bbd230387be870bda39f677339238ec2ab7e9466a8a84ea32daf058a2ed9677b`.

[Full metadata and provenance](data-plan.json)
