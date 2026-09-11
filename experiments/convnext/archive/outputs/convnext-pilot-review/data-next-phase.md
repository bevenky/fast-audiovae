# Data for the next decoder comparison

The clarified policy restores the previously downloaded multilingual and expressive training data. **We do not need another download before the next comparison.** Audio used by a discarded model may be used in a new experiment. A continuation of the current student still excludes its own previously scored windows. Both comparison arms may use the same ordered examples once each. Validation and test recordings remain excluded in every case.

A live, read-only Runpod manifest audit found **481.688 source hours across 169,994 existing recordings** available for continuing the current student. This conservatively excludes all 11,040 recordings used by its completed pilot, all held-out identities and the small fitted diagnostic. An independent new initialization has 501.734 source hours available. All referenced audio files exist. These are file durations, not yet allocated scored-window hours; short tails, masks and exact interval accounting still apply. [Detailed audit](data-reuse-availability.json).

## Language coverage

All 22 scheduled Indic languages are available under the clarified policy. The current continuation pool includes 41.56 hours of IndicVoices alone, with additional Indic speech in FLEURS. Bodo, Dogri, Konkani, Kashmiri, Maithili, Manipuri, Sanskrit and Santali each have approximately two hours; Odia has 3.98 hours across sources. Their earlier absence was caused by excluding consumption from older experiments, not by missing downloads.

| Other requested coverage | Available source hours for the current continuation |
|---|---:|
| English | 88.99 |
| Mandarin | 3.43 |
| Cantonese | 3.43 |
| Arabic | 3.21 |
| Latin American Spanish | 3.43 |
| Brazilian Portuguese | 3.43 |
| French | 3.43 |
| Japanese, including expressive recordings | 5.44 |
| German, including expressive recordings | 5.60 |

Training availability does not establish validation coverage. The existing 84-recording development panel still lacks nine Indic languages, Mandarin, Cantonese, Arabic and German. Add a separately versioned panel from eligible held-out sources before making claims about those groups. Do not move any record into validation after training an arm on it. The existing panel remains useful for the immediate matched comparison.

## Explicit vocal events

These counts exclude held-out source and known contributor identities and the current pilot. Durations include pauses or speech within each file; labels can overlap, so their seconds must not simply be added into an event quota.

| Event label | Available training files | Whole-file seconds |
|---|---:|---:|
| Laughter | 432 | 1,805.51 |
| Crying and sobbing | 15 | 143.70 |
| Giggle | 45 | 124.86 |
| Screaming | 136 | 489.12 |
| Shout | 15 | 41.62 |
| Human whistling | 14 | 114.84 |
| German whisper style | 260 | 1,153.62 |
| New Whispering supplement | 12 | 68.10 |
| New Breathing supplement | 12 | 51.71 |
| New Yell supplement | 2 | 7.32 |

The supplement's eight development recordings remain separate. Crying and giggling now have training examples again, but the current development panel still has neither. Sadness and amusement labels are not substitutes for these events. Yell remains distinct from Shout.

## Proposed 500-update comparison allocation

Prepare one deterministic manifest for both arms, excluding the parent student's 32,275 scored intervals. At the current batch size, 500 updates would expose roughly 9.9 scored hours per arm if the observed average valid crop length persists; the full-crop upper bound is 11.38 hours. Allocate by actual valid samples:

- **30% English speech.**
- **30% Indic speech**, with a guaranteed initial allocation to every one of the 22 languages and balanced remainder.
- **35% other speech**, explicitly including Mandarin, Cantonese, Arabic, Latin American Spanish, Brazilian Portuguese, French, Japanese and German.
- **5% verified vocal events**, capped at the actual available unique valid windows. Include the rare crying, giggling, shouting and human-whistling classes before filling the remainder with laughter and screaming. Never repeat a scarce file to fill a percentage. Disclose any shortfall and reallocate it to a named speech bucket before freezing the manifest.

These are exclusive source allocations. Quiet intervals, breaths, low volume and speech-to-silence transitions are additional overlapping monitoring categories. Measure them on teacher targets; do not claim coverage from an emotion or language label. Keep complete causal context, score each selected interval once per arm, and preserve the same order, tail handling and exposure journal in both arms. Record each arm as an independent branch of the same parent, rather than combining their ledgers and erroneously excluding the second arm's matched data.

## Preparation issues

The supplement files named `versions/v2/train.jsonl` and `dev.jsonl` contain concatenated, pretty-printed JSON objects. A normal `json.loads(line)` reader fails immediately on the first line, `{`. This has now been repaired in a separate Runpod version, `versions/v3-jsonl`, containing genuine one-object-per-line JSONL. The standard `load_manifest` and `validate_manifest` checks pass. All 26 training and eight development records, their order and every field are preserved; original and prepared audio hashes were verified for all 34 files. Version 2 remains unchanged. Use version 3 for the next preparation. [Repair evidence](expressive-manifest-repair.json).

The three FLEURS development languages Spanish, Portuguese and French share language-wide `unknown-session-group` placeholders with their training configurations. Those placeholders do not identify real sessions. This audit excluded actual source, hash, parent, known speaker and known session overlaps while disclosing the placeholders. Their true speaker independence remains unverified. Do not implement an exclusion that accidentally removes every training record in these languages, or claim that ignoring a placeholder proves speaker separation.

No new audio was acquired, decoded or trained during this audit. The source files remain on Runpod. The next step is preparing and validating this bounded comparison manifest, not acquiring another broad corpus.
