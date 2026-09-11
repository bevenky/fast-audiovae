# Language, nonverbal and quiet-audio coverage

The prepared 20-hour pilot can test whether the corrected training recipe generalizes across a useful speech mixture. It cannot establish coverage of all requested languages and vocal events. The new fixed development panel improves visibility into those gaps without putting development audio into training.

These results come from the live Runpod manifests and files. The source audio remains on Runpod. Six existing original benchmark WAVs were uploaded from the Mac to add Latin American Spanish, Brazilian Portuguese and French. Their bytes match the original benchmark manifest; no model reconstructions were used as source audio.

## What is actually available

| Scope | Available material | What it establishes |
|---|---:|---|
| Previously assembled training corpus | 510.93 h, 183,802 files | Included 102 FLEURS configurations and all 22 scheduled Indic languages before the previous run consumed data. |
| Whole untouched training utterances before new reserves | 139.17 h | Usable capacity is concentrated in 16 identified languages plus unspecified-language nonverbals. |
| Prepared new pilot | 20.007 scored h, 32,275 windows, 11,040 source files | 13 Indic languages, English, German and Japanese; 2.4 h of unspecified-language EmoGator material. |
| New speaker-separated development reserve | 4.81 h, 1,468 files | English and the same 13 Indic languages. |
| Sealed reserve, left untouched | 4.39 h, 1,310 files | English and the same 13 Indic languages. |
| New fixed development panel | 84 files, 167 beginning/interior windows | 18 identified languages plus unspecified-language nonverbals, including 15 explicitly labeled event files. |

The pilot contains 10.0005 h LibriSpeech, 5.0048 h FLEURS, 1.0005 h IndicVoices, 2.4001 h EmoGator, 0.8004 h CREMA-D, 0.4006 h JVNV and 0.4002 h Thorsten emotional audio. Source-file duration is slightly larger than scored duration because unscored tails are not counted as training exposure.

The 13 Indic languages are Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Punjabi, Sindhi, Tamil, Telugu and Urdu. Bodo, Dogri, Konkani, Kashmiri, Maithili, Manipuri, Odia, Sanskrit and Santali lack untouched training utterances in this assembled corpus. They also lack coverage in the new panel. Chinese and Arabic remain panel gaps. Spanish, Portuguese and French are present in the panel but absent from this new pilot's training mixture. German is in pilot training, but no existing German development recording was available; the Thorsten dev manifest is empty.

## Nonverbal coverage needs explicit labels

| Event | New pilot, verified file labels | New fixed dev panel | Gap |
|---|---:|---:|---|
| Laughing | 0 | 6 files, 22.58 s total source duration | Existing labeled training files were consumed by the previous run. |
| Screaming | 0 | 7 files, 9.93 s | Screaming does not establish ordinary shouting coverage. |
| Human whistling | 0 | 2 files, 34.04 s | Existing labeled training files were consumed. |
| Crying/sobbing | 0 | 0 | Needs fresh train and dev examples. |
| Giggling | 0 | 0 | Laughter is not a separate giggling test. |
| Shouting | 0 | 0 | Needs fresh train and dev examples. |
| Whispering | 39 files, 160.26 s total source duration | 0 | Training comes from one German speaker; held-out coverage is missing. |

The pilot also contains emotion-labeled nonverbal bursts and Japanese verbal/nonverbal material. Those are useful, but sadness is not a verified crying event and amusement is not a verified giggling event. File durations above include any pauses or speech within those files and must not be presented as pure event duration. Every old pilot window had `condition: null`; event-specific monitoring cannot be inferred from that field alone.

The saved FSD selection already acquired every qualifying clip under its current event, license, vote and co-label filters. The one unprepared selected ID was quarantined as an exact audio duplicate. Reusing those files would violate the no-repeat policy.

A metadata-only extension to the same pinned selection finds **96 fresh whispering and 176 breathing training candidates**, plus **7 whispering and 5 breathing development candidates**. These are candidate files, not acquired or audio-validated hours. No downloads were started. This extension does not fill crying, giggling, shouting or whistling gaps.

## The new panel and quiet checks

Selection used deterministic source metadata before reading signal levels or scoring any model. It includes three utterances per available Indic language except Gujarati, four English, ten Japanese, two each Spanish/Portuguese/French, twelve generic emotional nonverbals and all fifteen explicit dev event files. Gujarati has only one eligible development utterance after excluding the prior sentinel and shared known identities, so its result is a diagnostic rather than a language-level estimate.

All 84 source files passed exact SHA-256, finite mono 16 kHz decoding and sample-count checks. The panel contains 9.81 minutes of source audio and 5.72 scored minutes in its prescribed windows. After freezing selection, input-level measurements found 11.39 seconds below -80 dBFS and another 44.97 seconds between -80 and -60 dBFS. These are source-input measurements; the frozen teacher's 48 kHz quiet-window coverage must still be measured when targets are prepared.

The panel has no shared source IDs, file hashes, parent recordings or known speaker IDs with old training, the new pilot, or the fitted diagnostic/sentinel sets. It also has no shared session IDs with the new pilot. Historical FLEURS Spanish/Portuguese/French test files share three language-wide **unknown-session placeholders** with old training. Actual speaker/session identity is unavailable; those placeholders were preserved and disclosed, not renamed to manufacture separation. The original filenames and benchmark exclusions remain recorded.

This is development data that may have informed earlier audits, not a new sealed test. The sealed reserve was not used or scored. Acoustic duplicate detection beyond recorded identities and hashes remains unverified.

## What the next run should demonstrate

Report speech, quiet intervals and explicit vocal events separately, retaining per-language values. Check both residual noise and loss of real quiet sounds. Compare amplitude, waveform correlation where meaningful, and quiet residual errors at the prescribed positions, with no fitted gain or time shift. Inspect failures at speech-to-pause transitions and streaming boundaries.

The current run can validate the recipe over this declared scope. A successful aggregate score must not be reported as validation of all 22 Indic languages, Chinese/Arabic or the missing vocal-event classes. Fresh acquisition and a larger untouched evaluation panel are required before making those claims.

Evidence: [live coverage audit](coverage-audit.json), [fixed panel readiness](dev-panel-ready.json), [fixed panel files](dev-panel-v1/ready.json), [expressive candidate metadata](expressive-topup-candidates.json).
