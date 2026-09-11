# Existing Runpod corpus inventory

Live metadata audit, 2026-09-10. No audio was decoded, model loaded, training started, files deleted or audio hashes reread. The audit read manifests and checked file existence/nonzero size. Full machine-readable receipts are in [dataset-audit.json](dataset-audit.json).

| Inventory | Distinct sources | Manifest hours |
| --- | ---: | ---: |
| Master corpus, train and dev | 186,641 | 513.445 |
| Master train split | 183,802 | 510.927 |
| Master dev split | 2,839 | 2.518 |
| Master train after known R2/R9 reserved source, file-hash and parent exclusions | 181,008 | 501.698 |
| Earlier bootstrap outside the master | 230 | 0.764 |
| Later expressive topup outside the master | 282 | 0.289 |

All 186,641 master audio paths existed and were nonempty, totaling 94,241,948,026 referenced bytes. The additional 512 paths also existed and were nonempty. This confirms that the large downloaded corpus remains available, not merely a requested download quota. It does not claim every record is already teacher-cached or that audio bytes were newly revalidated. Additional inventories must pass the same reservation and deduplication merge before their hours are added to an eligible training total.

The authoritative master is `/workspace/fast-audiovae-convnext-20260908-r1/expanded-pilot/corpus/source-manifest.jsonl`, SHA256 `198ed7aa345db0675c8db3382b9856a31857a66b6a17c62ed9d581d781f5c8fd`. Its saved readiness receipt independently records 510.927 training hours, 183,802 training sources and original audio-byte/header verification. Current manifest arithmetic agrees. It includes 340.201 hours of FLEURS, 100.003 hours of LibriSpeech, 44.037 hours of IndicVoices and about 26.686 hours of other expressive material. GigaSpeech is not in this assembled corpus.

Training-manifest coverage includes all 102 FLEURS configurations and all 22 scheduled Indic languages. The original IndicVoices allocation is 2.000–2.005 hours per language; after the known reservation exclusion it is approximately 1.805–2.005 hours per language, with all 22 retained. English, Spanish `es_419`, Portuguese `pt_br`, Mandarin `cmn_hans_cn`, Japanese, French and Arabic configurations are present. These locale tags identify dataset configurations; this audit does not independently establish each speaker's nationality, accent or geographic origin. Raw language fields have 128 tags; the readiness normalization has 111 tags including `und`, so neither count should be advertised as that many independently identified languages. Speaker, emotional style, held-out language coverage and event-interval coverage require their own metadata checks; the presence of expressive source labels is not proof of a particular event throughout a recording.

The 501.698-hour result excludes the listed R2/R9 reserved manifests plus the master dev split using exact source identity, declared file hash and parent recording identity. It is a preliminary eligible inventory, not a newly sealed training split or a proof that every later historical evaluation source has been excluded. The user authorized reuse of old training data for the new experiment; genuine held-out data should remain reserved. Previously evaluated development panels are still useful for regression diagnostics but are not fresh final holdouts.

Do not exclude unknown session placeholders as though they were verified identities. A deliberately broad speaker/session exclusion reduced the apparent total to 468.411 hours and removed entire FLEURS configurations. Inspection showed matches such as `fleurs:unknown-session-group:cmn_hans_cn`, including different official source splits. The same issue affects Arabic, Spanish, Portuguese and French. That number is not a valid clean-corpus count. Finalize an explicit policy for missing speaker/session IDs, retain source/parent/file checks, and state when speaker separation cannot be verified.

Storage is the immediate operational limit: `/workspace` has about 182 MiB free, `/tmp` about 4.2 GiB and `/dev/shm` about 45 GiB. The last location is volatile RAM, not durable checkpoint storage. Additional complete checkpoints, optimizer states and replacement-stage caches need an explicit storage plan before training. No existing artifacts were removed by this audit.
