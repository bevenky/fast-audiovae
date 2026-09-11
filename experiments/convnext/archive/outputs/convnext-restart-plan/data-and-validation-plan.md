# Data and validation plan for the repaired decoder pilot

**Recommendation: fresh student and optimizer weights, the same frozen AudioVAE2 teacher, and a 20-hour pilot drawn only from previously unscored material.** Preserve the step-1,000 and step-10,000 checkpoints as frozen comparisons. Do not restart over the previously consumed 371.70 hours. This plan does not authorize downloads or training.

The loss and real-waveform learning gates belong to the accompanying restart plan. This document specifies the data and validation corrections. Capacity figures come from saved manifests and exact sampler reconstruction, without audio scans: [capacity evidence](unused-source-capacity.json).

## 1. Freeze a global consumption and exclusion ledger

Record the old 576,000 scored windows by immutable source identity and 16 kHz sample interval. Mark all 230 earlier bootstrap training utterances unavailable, conservatively covering its sampling-with-replacement history. Also include canonical parent/session identities and original/prepared hashes; changing a filename or encoding never makes old audio new.

For main training, allow an unused window in an already touched utterance only if its scored interval does not overlap the ledger. Real causal context may overlap old intervals, but is not scored or counted as new exposure. For the proposed pilot, wholly untouched utterances alone provide sufficient capacity, so prefer those.

For development or sealed testing, exclude the **entire** previously touched utterance and all known speakers, sessions and recording copies associated with training. Unused tails from previously trained utterances are not held-out data. Preserve the existing 80 legacy evaluation exclusions. Publish the complete selected window list, quotas and ledger hash before training; checkpoint the committed cursor only after each successful optimizer update.

## 2. Make the diagnostic repeat exception explicit

After approval, use a separate tiny set of 16–32 fresh real-teacher examples to prove that the full decoder can fit waveform amplitude and phase under the corrected objective. Repeating those fixed examples is a deliberate diagnostic exception to the no-repeat rule. Reserve their complete utterances from the main pilot and final evaluation.

Discard diagnostic weights before the main fresh-start pilot. Ordinary validation repeatedly evaluates the same held-out examples without updates; it is not repeated training. Do not silently run fresh-start and warm-start arms on the same training windows.

## 3. Reserve genuinely untouched validation first

| Existing source | Eligible reserve found | Use |
|---|---:|---|
| LibriSpeech | 19 never-trained speakers; 2,174 utterances; 7.737 h | Reserve all those speakers. Split them deterministically into development and sealed-test groups. |
| IndicVoices | 567 never-trained speakers; 604 utterances; 1.465 h across 13 languages | Reserve all those speaker groups. Keep speaker-disjoint development/test subsets. |
| Existing expressive dev | Original actor/contributor-separated development recordings | Retain as development diagnostics with balanced source weights. These have already informed the audit, so are not a new blind test. |

Gujarati has only four genuinely unseen-speaker utterances totaling 38.08 seconds. That supports a small diagnostic, not a reliable language-quality estimate. Nine Indic languages have no untouched training utterance left in the assembled corpus: Bodo, Dogri, Konkani, Kashmiri, Maithili, Manipuri, Odia, Sanskrit and Santali.

An initial **64-utterance fixed development panel** can contain 16 LibriSpeech utterances from at least eight reserved speakers, two utterances for each of the 13 available Indic languages, and 22 expressive dev utterances. Select the panel by frozen metadata rules and then check audio validity; do not select only clips the student already handles well. If a claimed expressive event is absent from dev labels, report the gap instead of substituting an emotion label.

Compare the teacher, silence control, old student and new student on identical examples. Record teacher distance separately from optional original-reference distance, and report source/language means alongside aggregate results. Evaluate the beginning and a valid interior region where available, full utterances and actual streaming boundaries. Do not fit per-clip gains or time shifts. A balanced panel prevents 1,530 EmoGator examples from dominating the result.

## 4. Train the bounded 20-hour pilot without replacement

These are **valid scored-hour quotas**, not fixed clip-count probabilities.

| Exclusive source bucket | Pilot hours | Share | Available wholly untouched audio before new reserves |
|---|---:|---:|---:|
| LibriSpeech | 10.0 | 50% | 98.02 h |
| FLEURS | 5.0 | 25% | 17.81 h |
| IndicVoices | 1.0 | 5% | 3.411 h |
| EmoGator | 2.4 | 12% | 12.45 h |
| CREMA-D | 0.8 | 4% | 4.31 h |
| JVNV | 0.4 | 2% | 1.37 h |
| Thorsten emotional | 0.4 | 2% | 1.80 h |
| **Total** | **20.0** | **100%** | |

Reserving all identified unseen-speaker LibriSpeech and IndicVoices pools still leaves ample capacity for these pilot quotas. The exact window allocator must confirm short-tail eligibility and diagnostic exclusions before launch; it must stop rather than silently lower a quota.

FLEURS allocation: 0.25 h each for Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Punjabi, Sindhi, Tamil, Telugu and Urdu; English 0.60 h; German 0.35 h; Japanese 0.80 h. IndicVoices allocation: at least three minutes from each of those 13 languages, then allocate the remaining 21 minutes within verified residual capacity. Together these provide 16 identified speech languages plus unspecified-language nonverbal material. This is a waveform-learning pilot, not complete multilingual qualification.

Within a bucket, interleave actual speakers where IDs exist and cap long runs from one recording. Use valid-sample deficit scheduling to maintain the declared source proportions over each roughly one-hour exposure block. Do not give an unspecified-language nonverbal bucket only one turn among 111 labels. Track achieved source, language, speaker, duration and verified condition exposure. Do not increase step count by wrapping the sampler: stop when the fixed pilot window list is exhausted or an earlier learning gate fails.

EmoGator supplies emotion-labeled nonverbal bursts; CREMA-D supplies intended emotions; Thorsten has an explicit whisper style; JVNV supplies generic annotated nonverbal intervals. These labels do not establish separate giggling, crying or shouting categories. All existing training FSD vocal and human-whistling clips were already consumed. Verified fresh examples of those events require new sources or new qualifying IDs before a subsequent main phase.

## 5. Replenish and validate before broad training

The current 340.20 FLEURS hours are a selected subset, not its full official train corpus. Saved pinned TSV metadata suggests about **472.02 additional eligible hours** across configurations, before new exclusions and file verification. Useful nominal headroom includes ES-419 5.24 h, PT-BR 4.15 h, French 3.91 h, Mandarin 6.16 h, Cantonese 3.50 h and Hindi 3.07 h. These are planning estimates, not acquired usable hours.

Seven configurations have zero estimated headroom under the existing filters: Afrikaans, Arabic, Filipino, Icelandic, Mongolian, Oromo and Odia. Japanese has only 0.318 h. Do not promise fresh coverage of all 102 from that same filtered release alone. Use a new acquisition root, retain the pinned revision, and exclude all already acquired source identities from top-up downloads; the unused existing corpus stays separately available. Indic top-ups must also exclude prior full-recording sessions because unpublished chunk offsets cannot prove disjointness.

For representative multilingual development, acquire a small fixed subset of **official FLEURS dev**, without putting it into training. The original paper reports different speakers between train and dev/test; individual speaker IDs are unavailable, so label this as publisher-reported separation, not independently verified speaker identity. [FLEURS paper, section 4.2](https://arxiv.org/html/2205.12446#S4.SS2). Preserve the true official split and its evidence in the importer rather than inventing speaker IDs.

Before any larger phase, require fresh training coverage and held-out coverage for all 22 Indic languages, Latin American Spanish, Brazilian Portuguese, Mandarin/Cantonese, Japanese, French and Arabic, then the remaining requested languages. Replenish exhausted languages and verified vocal-event classes from sources with reviewed commercial-compatible terms. GigaSpeech remains excluded. Dataset notices and original byte provenance remain intact.

Until replenishment and storage capacity are verified, the only executable training budget in this proposal is the bounded 20-hour pilot. Do not advertise that pilot as a replacement codec or extend it automatically to consume all remaining 139 hours.

