52 additional held-out recordings are downloaded and validated on Runpod. All 13 requested language gaps have four recordings each. The audio totals 8.09 minutes and the complete staging directory occupies 17.27 MiB. No audio was downloaded to the Mac.

| Cohort | Recordings | Audio seconds | Published speaker identity |
|---|---:|---:|---|
| Nine missing Indic languages | 36 | 307.19 | 36 distinct speakers and sessions |
| Mandarin Chinese | 4 | 43.04 | Unknown |
| Arabic | 4 | 43.18 | Unknown |
| German | 4 | 45.66 | Unknown |
| Cantonese | 4 | 46.14 | Unknown |
| Total | 52 | 485.21 | 36 known, 16 unknown |

The Indic additions cover Bodo, Dogri, Konkani, Kashmiri, Maithili, Manipuri, Odia, Sanskrit and Santali. They come from the official IndicVoices `valid` partition. The other four languages come from official FLEURS `validation` partitions. Both releases publish CC-BY-4.0 licenses; attribution and pinned source revisions are retained in each recording's receipt. [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices/blob/main/README.md), [FLEURS](https://huggingface.co/datasets/google/fleurs/blob/main/README.md).

The validation checked every source used by all 320,000 planned optimization windows and all 512 calibration windows, including future unused windows. It also checked the existing held-out inventory across 15 manifest snapshots. All new source-byte hashes, recording identities, published speaker/session identities, receipts and decoded lengths passed. The exclusion manifests remained unchanged.

The original delivered audio bytes are retained without gain changes, normalization, enhancement or resampling. This acquisition did not prepare teacher targets, run quality benchmarks, start training, or change an active data plan.

Four recordings per language establish a small coverage check. They do not support a reliable language-level quality ranking. FLEURS does not publish actual speaker/session IDs, so its recordings are source/hash disjoint but unseen-speaker generalization is unverified. The existing explicitly unknown session placeholders remain in the metadata. Identity checks do not claim acoustic-fingerprint detection of unknown re-encodings.

The separate appendix is sealed as `language-expanded-manifest.jsonl` on Runpod, SHA256 `fa33aafe748f612b877d51277f453705495c19e9bc37f0017ba08d502868e152`. It preserves the earlier 36-row Indic and 48-row combined snapshots. No further rows are pending.

[Manifest](language-manifest.jsonl) · [Ready metadata](language-ready.json) · [Full validation evidence](language-validation.json)
