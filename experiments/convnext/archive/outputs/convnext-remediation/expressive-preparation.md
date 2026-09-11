# Expressive data preparation

The supplement is verified and staged on Runpod. It has not entered training and no model, optimizer, normalization, active schedule or teacher cache was changed.

| Verified training supply | Clips | Once-only scored audio |
|---|---:|---:|
| EmoGator emotional nonverbal bursts |28,871|15.048 hours|
| Whispering |70|251.52 seconds|
| Breathing |161|456.73 seconds|
| Yell |51|330.02 seconds|
| Total |29,153|15.337 hours|

The supplement contains **29,429 nonoverlapping candidate windows**. Each window has at most 64 latent frames, a minimum of 3,040 valid input samples, and requires the existing 29-frame context contract. There are no unusably short source files. Whole-recording supply is 15.33686 hours; valid scored supply is 15.33658 hours.

EmoGator was downloaded already but conservatively omitted by the original planner because its language is unknown and its annotations describe intended emotions rather than explicit vocal actions. The official [EmoGator release](https://github.com/fredbuhl/EmoGator) documents non-speech vocal bursts and contributor identifiers. Its files are now staged under a separate `emotional_nonverbal` condition while retaining the 30 original emotion categories. This does not relabel sadness as crying, amusement as giggling, or anger as shouting.

The fresh individual FSD recordings were acquired directly on Runpod, with four bounded download workers. The [FSD50K release](https://zenodo.org/records/4060432) supplies clip licenses and human annotation votes. Existing strict checks were retained: original train partition, approved CC0/CC-BY per-file licenses, two predominant-presence votes, no negative presence vote, and no non-vocal background co-label. Yell remains distinct from Shout. Counts describe windows from labeled recordings, not dense event annotations.

Three additional breathing recordings, totaling **17.83 seconds**, were reserved from two contributors absent from the full current training and calibration plans. No unused whispering or Yell contributor was independent of the already planned training contributors, so those recordings were not reassigned to validation. Existing reserved validation remains intact. One fresh screaming candidate was rejected because its original bytes duplicated an excluded recording. No new verified crying, giggling, shouting or human-whistle recordings were added; these gaps remain open.

Every accepted file passed prepared-byte SHA-256, full finite/nonzero mono16k decode, exact sample count and preparation receipt checks. Existing EmoGator receipt identities were checked against the pinned original Git tree, and the original license and README bytes were verified. Fresh FSD original bytes were checked against pinned upstream content hashes. All 70,353 complete current planned source files and all historical heldouts were excluded where required; final window checks include 5,634 heldout sources and 70,369 current/fitted source exclusions. FSD uploader grouping is a conservative source guard, not proof of physical speaker identity.

## Authoritative handoff

Use `ready-v2.json` under `/workspace/fast-audiovae-convnext-20260909-r9/remediation/expressive/`, mirrored in [expressive-handoff.json](expressive-handoff.json). It pins:

- `train-candidates-v2.jsonl`
- `fresh-dev-v2.jsonl`
- `candidate-windows-v2.jsonl.gz`, read with `gzip.open(path, "rt")`
- source IDs, condition maps and new contributor reservations

The first manifest publication accidentally contained pretty multi-line JSON object streams. The window loader rejected them before any training use. Those original files are preserved but explicitly superseded. Both corrected manifests pass the actual loader round-trip, and the full candidate interval checks pass. Only the v2 paths above are accepted.

## Feasible continuation

At a step-3,000 boundary, the original schedule has **144.763 scored hours** left, including **0.803 hours** of explicit event material. If all that unconsumed material is retained, a future 5% broad expressive mixture needs approximately **6.774 additional hours**: all **17.30 targeted minutes** plus **6.486 generic nonverbal hours**. The staged capacity is sufficient. This arithmetic does not establish 5% verified crying/whistling coverage, and an actual continuation must preserve the engine state, global step and exact consumed-interval history.

Stage storage is **391.3 MiB**, below the 400 MiB allocation. Free space at publication was **5.50 GiB**, above the 4 GiB reserve. All audio remains on Runpod. No new checkpoint or audio was downloaded to the Mac.
