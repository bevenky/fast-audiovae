# Training monitoring and data remediation

The reporting changes are active and the trained student has continued through a versioned data handoff at step 5,900. Decoder architecture, loss weights, calibration and optimizer state are preserved. Every remaining original window stays in its original relative order, with additional expressive windows interleaved.

## Applied now

- The existing TensorBoard address now serves a corrected view built from saved metrics. `loss/reconstruction_only` means waveform plus mel. Separate plots show waveform, mel, adversarial and feature-matching losses, together with scheduled and measured component-gradient norm fractions.
- Quiet residual RMS, the repeating 480-sample residual, peak amplitude and full-scale sample counts are visible. A reporting check combines reconstruction acceptance with peaks within full scale. It does not gate or modify training.
- A separate observer preserves milestone checkpoints with the actual step, SHA-256, optimizer state and exposure cursor. It retains at most two of its snapshots; the earlier step 1,700 audit snapshot and sealed step 5,900 parent checkpoint remain separate. Interrupted publication and pruning are recoverable from checkpoint receipts. The trainer and its original logs are untouched.
- The exact step 3,000 checkpoint was preserved and evaluated on CPU against the unchanged 84-recording, 167-crop panel. New validation cohorts have separate chart prefixes and do not enter training.

The reporting tests cover loss-label semantics, scheduled versus measured gradient shares, peak acceptance without modifying the old gate, atomic checkpoint replacement, recovery before observer-state persistence, interrupted checkpoint publication and interrupted pruning. Three test functions passed. An independent review found and then verified the fix for an observer-interruption durability issue.

A live verification found training at update 4,407 and the observer at 4,404. Both event and language appendix charts were visible, with one displayed run and no misleading `train/total` tag. Free space was 5.32 GiB. A later strict checkpoint audit at step 4,700 verified the optimizer, calibrated statistics and unchanged training implementations.

## Original fixed-panel review

These are teacher reconstruction diagnostics, not PESQ, UTMOS, DNSMOS or listening ratings. Both checkpoints below were evaluated on CPU using the same fixed teacher targets.

Correlation here is the uncentered waveform cosine used by the existing fixed-panel checks.

| Measure | Step 1,700 | Step 3,000 |
|---|---:|---:|
| Speech waveform correlation | 0.824 | **0.900** |
| All active audio correlation | 0.701 | **0.793** |
| Normalized waveform error | 0.338 | **0.272** |
| Teacher mel error | 1.527 | **1.367** |
| Mean quiet residual RMS | 0.000539 | **0.000338** |
| Median repeating residual RMS | 0.000268 | **0.000187** |
| Highest absolute output peak | 1.398 | **1.348** |
| Samples at or above full scale | 75 | **112** |
| Crops with full-scale samples | 2 | **3** |

Waveform error improves 19.6%, mel error 10.4%, quiet residual 37.2% and the repeating residual 30.2%. Laughter correlation improves from 0.589 to 0.709; screaming from 0.066 to 0.248; human whistling from 0.068 to 0.157. The latter conditions remain weak.

All 1,540 quiet windows still fail the current strict checks. Two of 165 active crops now reach 0.99 correlation. Three laughter crops from two recordings exceed full scale; their teacher targets remain below full scale. The largest peak fell, but affected-sample count rose, so peak behavior is not fixed. This supports continued training and explicit monitoring, rather than declaring acceptance or resetting the model.

The evaluation performed zero optimizer updates, and model weights and frozen targets were unchanged by it. These improvements occurred before the staged supplemental training audio was used.

## Additional validation

The seven existing reserved whisper/breathing recordings passed source-byte hashes, decoded sample counts and identity exclusions against all 320,512 planned optimization/calibration windows. They provide 12 crops and 50.59 seconds of source audio, representing four uploader groups with unknown physical speaker identities. Labels identify recordings, not dense event intervals or a new listening assessment.

| New event panel | Correlation at 1,700 | Correlation at 3,000 |
|---|---:|---:|
| Whispering | 0.406 | **0.515** |
| Breathing | 0.384 | **0.431** |

No full-scale samples occurred in this event panel, but all 167 quiet windows still fail. Teacher targets were prepared once on CPU and reused. These results appear under `appendix/`, separate from the original `quality/` aggregate.

The missing-language acquisition produced 52 official held-out recordings, four each for nine additional Indic languages plus Mandarin, Cantonese, Arabic and German. The 36 Indic recordings have distinct known speakers and sessions. The 16 FLEURS recordings lack published speaker/session identities, so their independence is established at the recording/hash level and is not claimed at the speaker level. The original panel plus the Indic additions covers all 22 scheduled Indian languages.

The step 3,000 language evaluation completed 102 crops from all 52 sources. Active waveform correlation averages **0.9115**, with **0.9215** across the nine additional Indic groups. All 1,629 quiet windows still fail; one Konkani crop has 16 full-scale samples and peak 1.137. These results are separately visible under `language_appendix/`. Four sources per language are a coverage check, not a reliable ranking. These sources remain reserved from training.

## Verified training data, staged separately

| Usable supplement | Once-only scored audio |
|---|---:|
| Existing EmoGator emotional nonverbal bursts | 15.048 hours |
| Fresh whispering | 4.192 minutes |
| Fresh breathing | 7.612 minutes |
| Fresh Yell recordings | 5.500 minutes |

The 29,153 accepted training sources yield 29,429 nonoverlapping candidate windows, totaling 15.337 scored hours. EmoGator was already downloaded but excluded because its language was unknown and its labels describe emotions rather than explicit actions. It now retains the separate `emotional_nonverbal` category and original emotion annotations. Sadness is not relabeled crying, nor anger shouting.

Every accepted source passed byte-hash, full finite/nonzero mono16k decoding, exact length and preparation-provenance checks. Fresh FSD files passed the existing per-file license, annotation and upstream-hash filters. Three additional breathing recordings were reserved from two new contributors before any training selection. One screaming candidate duplicated excluded audio and was rejected.

The original pretty-printed manifest publication was rejected by the window loader before training use. It is preserved but superseded. **Only the canonical v2 manifests and the authoritative handoff below are accepted.**

No new verified crying, giggling, shouting or human-whistle recordings were added in this bounded acquisition. Those named shortages remain open. The 15 hours of generic nonverbal material must not be represented as filling those specific labels.

## Completed training continuation

The parent stopped at a verified committed **step 5,900**: checkpoint, 188,800-window cursor, exposure journal and metrics all agreed. A failed boundary check would have resumed that same process. The preparation retained original data/source identities and all old/new heldout reservations. Only regenerable teacher target-cache files were evicted to recover approximately 2 GiB; source audio and model checkpoints were preserved.

The successor keeps **131,200 remaining original windows** and adds **7,655** supplemental windows. The added valid audio is **3.988 hours**, comprising all **17.30 minutes** of fresh whispering/breathing/yell data plus **3.700 hours** of generic emotional nonverbal audio. Broad expressive material is now **5.0002% of the remaining segment**, not 5% of the entire history and not 5% verified crying/whistling. The new global budget is **10,240 steps**, explicitly extending the old 10,000-step budget to preserve all remaining original data.

All original optimizer/discriminator/RNG and calibrated-normalization state was restored. The first two new updates completed and saved at **5,902**, with 64 segment windows consumed and exactly **5,402 discriminator updates**. Sustained training was then launched with exact resume from that cursor. No random-weight restart, repeated calibration or loss-policy change occurred.

The sustained run was subsequently verified at **step 5,982**, with **2,624** segment windows consumed and **5,482** discriminator updates. The trainer and observer were alive, the observer had reached the same step, and no training error was recorded. TensorBoard still displayed one run, with the new training-peak plots present and the misleading `train/total` absent. Available disk space was **5.72 GiB** at that check. These are a point-in-time status, not a promise of future completion.

The new runner logs raw teacher/student peaks and full-scale counts over actual scored training regions on the first new batch and every 100 updates. This uses the existing forward output and adds no extra model forward. A detached observation test confirms identical training state and RNG with or without reporting. It also supports a clean between-update pause request.

The continuation correctness suite passed **31 tests on Runpod with GPUs hidden**, including native Muon plus AdamW. Data planning tests passed 54 checks locally, and reporting/retention checks passed 24. These suites include overlapping cases, so their counts should not be summed into a single claim. New code remains uncommitted.

All audio and checkpoints remain on Runpod. Local outputs contain reports and metadata only. No commits or pushes were made.

## Evidence and dashboard

- [Focused training dashboard](https://34d6pb4ub5ldrz-8888.proxy.runpod.net/#scalars&tagFilter=%5E(loss%2F(teacher_waveform%7Cteacher_mel)%7Cquality%2Fall%2F(nonquiet_cosine_mean%7Cquiet_residual_rms_mean%7Cpeak_abs_max))%24)
- [Step 3,000 fixed-panel report](fixed-panel-step3000-summary.json)
- [Whisper/breathing validation](validation-appendix.md)
- [Missing-language acquisition and coverage](language-coverage.md)
- [Language source and split verification](language-validation.json)
- [Language reconstruction results](language-validation.md)
- [Quiet and peak audit](quiet-peak-audit.md)
- [Verified expressive preparation](expressive-preparation.md)
- [Authoritative expressive handoff](expressive-handoff.json)
- [Data-continuation contract](continuation-contract.md)
- [Committed step 5,900 handoff receipt](committed-handoff.json)
- [Verified sustained continuation status](continuation-live-status.json)
- [Scheduled step 5,000 quality](fixed-panel-step5000-summary.json)
- [Encoded-silence diagnostics](encoded-silence-audit.json)
- [Earlier runtime verification](runtime-status.json)

Retained step 3,000 checkpoint SHA-256: `027387045c28a60883d2696568156ad1dbaa971dbb4e6086be765a486363c8a2`.
