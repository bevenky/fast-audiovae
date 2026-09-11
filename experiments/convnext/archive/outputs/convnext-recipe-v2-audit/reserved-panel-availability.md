# Existing reserved validation availability

The existing dev/test inventory can add **7 recordings covering whispering and breathing** immediately. It cannot fill the remaining language or crying/giggling/shouting gaps without additional independently reserved data.

| Missing validation group | Existing eligible files | Whole-recording duration |
|---|---:|---:|
| Nine missing Indic languages |0|0|
| Chinese: Mandarin and Cantonese |0|0|
| Arabic |0|0|
| German |0|0|
| Crying |0|0|
| Giggling |0|0|
| Shouting |0|0|
| Whispering |4|30.90s|
| Breathing |3|19.69s|

The nine Indic gaps are Bodo, Dogri, Konkani, Kashmiri, Maithili, Manipuri, Odia, Sanskrit and Santali.

The audit considered **5,631 unique rows already declared dev/test** from the reserved/excluded plans, full source and candidate-dev manifests, and existing expressive dev manifests. A training row was never considered eligible, even if its optimization window lies in the future.

All seven available recordings have existing files and published FSD source labels. Their source IDs, recorded audio hashes, parent identities and available speaker/session identities are disjoint from every source in the complete 320,000-window optimization and 512-window calibration plans. They are also seven distinct files absent from the current 84-source panel. No requested zero-availability group was lost through an overlap rejection or a missing file; those groups have no declared reserved candidate.

**Proposed addition:** preserve the current 84-source panel and all historical scores unchanged, and add a separately reported 7-recording event appendix totaling 50.59 seconds. This gives 91 unique validation sources, while clearly keeping the new condition results separate from the existing trend aggregate.

The seven files lack known individual speaker identities. Source/hash/parent disjointness is verified from metadata, but speaker independence is not proven. Labels describe recordings rather than dense per-sample event intervals. Actual audio hashes, decoded lengths and the audible event content should be verified when the new panel is evaluated.

No files were selected into a new manifest, no audio was read or transferred, and no inference, downloads or training changes were made. One permission-review timeout occurred before the inventory command started; the permitted read-only retry succeeded.

