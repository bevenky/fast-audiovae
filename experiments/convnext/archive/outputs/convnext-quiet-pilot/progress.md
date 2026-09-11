# Quiet-audio correction and representative pilot

The quiet loss is implemented, but the tested 10% gradient allocation was **not accepted**. The next model uses a fresh initialization and varied, previously unused training windows, with quiet-window checks enabled. The current trained codec has not been declared fixed or ready for release.

## Latest observed progress

The run has completed **1,009 / 1,009 updates** and stopped for review. Final development mean nonquiet cosine is **0.52541**, normalized waveform error **0.48996**, and mel error **1.64926**. It is still improving, but no nonquiet crop reaches 0.99 and all 1,540 quiet windows still fail. Most active outputs are too quiet. See the [completed audit](../convnext-pilot-review/saved-metrics-audit.md) and [recipe audit](../convnext-pilot-review/recipe-audit.md). The saved `live-snapshot.json` is the historical update-433 snapshot, not current status.

The sections below preserve the launch and calibration history. The user has since clarified that training data may be reused across independent diagnostic experiments, while scored intervals must not repeat within one run. Development and test audio remain separate.

## First scheduled result

The first 200 updates completed in 223.2 seconds of training-loop time, consuming **3.978 hours across 6,400 distinct windows**. The original teacher state was verified unchanged afterward. This timing excludes startup and development evaluation and is not inference RTF.

Held-out mean nonquiet cosine increased from -0.00067 at initialization to **0.07288**. Mean normalized waveform error changed from 1.36167 to 1.30025. **No nonquiet crop meets 0.99, and all 1,540 quiet windows still fail.** Laughter cosine is 0.0423, screaming 0.0003 and whistling 0.0037. These are early reconstruction failures, not successful quality validation. The continuation has passed update 277 with 5.506 hours consumed. The actual GPU resume preserved a strictly consecutive 277-entry exposure journal, and the gradual mel introduction is active. The run is bounded to 1,009 total updates and will then stop for review. [Saved 200-update result](stage200-result.json).

## What the calibration established

Both arms started from the identical fitted diagnostic checkpoint and optimizer state and used the same 32 original teacher crops. Each ran 150 additional updates. No teacher inference or teacher updates occurred, and neither fitted model seeds the representative pilot.

| Fitted diagnostic measure | Control | 10% quiet contribution |
| --- | ---: | ---: |
| Mean quiet-window residual RMS | 0.0001560 | 0.0001432 |
| Mean normalized waveform error | 0.02594 | 0.03083 |
| Minimum nonquiet cosine | 0.99672 | 0.99597 |
| Quiet windows failing the strict check | 100 / 123 | 105 / 123 |

The candidate reduced mean quiet residual by **8.2%**, but increased waveform error by **18.9%**. The predeclared rule required at least a 10% quiet improvement, no more than a 10% waveform regression, and minimum nonquiet cosine 0.99. It failed the first two. Held-out sentinels did not select the loss weight and remained poor in both arms. Lower mean residual also did not translate into fewer failing quiet windows, which is another reason not to promote it. [Saved measurements](quiet-calibration-summary.json).

## What is running next

The bounded representative phase uses the published 20.007-hour plan: 32,275 unique scored windows from 11,040 untouched recordings. The original encoder and decoder teacher stay frozen; the student receives their raw 64-channel posterior means and 48 kHz targets. The teacher runs on complete utterances before crops are selected, with qualified batches and a bounded rolling cache.

The first launch is capped at 200 updates with batch size 32. The complete fixed plan has 1,009 updates, including a short final batch. The recipe starts with waveform supervision; mel share ramps from zero after update 250 to 25% by update 500. The rejected auxiliary quiet loss is disabled. No adversarial stage starts automatically. The existing architecture is unchanged, and none of the analysis losses adds CPU inference work.

Checkpoints preserve the model, Muon/AdamW states, normalization, RNG, curriculum and committed sample cursor. An exposure journal and exclusive run lock prevent silent replay after interruptions. The teacher cache is numerically qualified; floating-point batching can produce small differences when targets are regenerated, so this is not a claim of bit-identical future targets across every cache/resume pattern.

## Validation and remaining gaps

The fixed development panel contains **84 recordings, 167 prescribed windows and 18 identified speech languages**, plus unspecified-language nonverbals. It includes 6 explicit laughter, 7 screaming and 2 human-whistling examples. Selection was based on metadata before measuring the student. Every selected source file, decoded length and checksum was verified against exclusions.

The pilot itself has 13 Indic languages plus English, German and Japanese. Its 39 explicit whispering training files are useful, but it lacks fresh explicitly labeled laughter, crying, giggling, shouting and whistling files. The development panel still lacks explicit crying, giggling, shouting and whispering, Chinese/Arabic coverage, and nine scheduled Indic languages. The panel's Spanish and Portuguese examples are Latin American Spanish and Brazilian Portuguese. These small per-language counts support an early diagnostic, not complete language qualification. See the [coverage audit](coverage-findings.md).

Low-energy coverage is measured separately on the actual teacher targets during training. Logs count 20 ms RMS bins, exact-zero samples, quiet transitions, peaks/clipped samples, beginning crops and partial tails. This avoids assuming that an emotion label guarantees a vocal event or that an entire quiet recording covers speech-to-silence transitions. Whole-clip cosine and local quiet residual checks remain separate.

## Fresh expressive supplement

A separate supplemental set now contains **34 newly acquired, verified recordings**: 12 whispering, 12 breathing and 2 labeled Yell training clips (127.13 seconds total), plus 4 whispering, 3 breathing and 1 Yell development clip (57.73 seconds). Contributors were reserved for development before choosing training clips; all 16 training and 5 development contributor groups are disjoint. Original and prepared bytes and the pinned release hashes passed verification, with no clipping or too-short recordings. Yell remains its explicit source label and is not silently renamed Shout. The 31-clip first version is preserved.

The files were downloaded directly to Runpod using individual-file requests. They have **not been trained on or added to the frozen 84-recording panel**. They fill part of the next phase's quiet/expressive data needs; crying, giggling and shouting still require additional qualifying examples. [Supplement findings](expressive-topup-findings.md), [version-two readiness](expressive-topup-v2-ready.json).

## Validation of the implementation

The final pilot stack passed **135 CPU tests plus 9 subtests both locally and on Runpod**. Coverage includes exact legacy and quiet-enabled engine state resumes, causal crop accounting, masked context/padding, finite gradients, quiet noise/muting/inversion, immutable window order, interrupted-run exposure handling, frozen teacher caches and held-out source separation. These are implementation checks, not perceptual-quality scores.

About 12 GB of reproducible targets from the stopped older run were reclaimed. Original recordings, checkpoints, cache identity, qualification evidence and target inventory remain preserved. All new dataset processing and training occur on Runpod. Only existing local benchmark reference recordings were uploaded to fill part of the development panel.

All source changes remain on `Convnext`, uncommitted. Training progress is available in [TensorBoard](https://34d6pb4ub5ldrz-8888.proxy.runpod.net/#scalars&tagFilter=validation%2Fall).
