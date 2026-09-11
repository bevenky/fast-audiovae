# Data and calibration audit

The inspected data path and exposure accounting passed. The main limitations are rare-event exposure and incomplete heldout coverage, rather than repeated training windows or missing calibration.

This is a read-only audit of r9. No model inference, source edits, training changes or audio downloads were performed. The active run advanced during the audit; the journal snapshot ends at step 1,497, teacher-level coverage at 1,498, and the separately loaded CPU checkpoint at 1,600.

| Duration share | First 1,000 updates | Full 10,000-update plan | Calibration |
|---|---:|---:|---:|
| English |31.36%|31.29%|23.62%|
| Indic |30.80%|31.06%|25.79%|
| Other identified languages |37.26%|37.09%|48.42%|
| Dedicated explicit events |0.579%|0.563%|2.172%|
| Total distinct scored audio |20.578h|206.540h|20.51min|

Other-language recordings include explicitly labeled Japanese nonverbal/mixed material; that category is not entirely ordinary speech. Calibration audio is separate training data and receives no optimizer updates.

## Verified

- Both immutable data-plan checksums loaded successfully. All 320,512 selected intervals passed combined source/hash/parent identity and scored-overlap validation, including 320,000 optimization windows and 512 calibration windows. Existing heldout/source exclusions remain in place.
- All 1,497 saved journal entries matched their exact 32-window batch, sequential step/cursor, sampler identity and previous-record hash. Same-file crops can occur repeatedly, but their scored intervals do not repeat. Shared causal context is intentional.
- Exact valid input/output sample accounting was checked: output length is 3 times input length. No planned window is shorter than 9,120 valid output samples. Partial tails remain included when they meet that minimum. First 1,000 updates contain 20.52% partial windows; the new reconstruction pools actual valid samples/mel elements rather than giving every short tail a full example weight.
- Calibration ran after update 500:512 windows, 16 batches each pass, two passes, zero optimizer updates. Both passes hashed identical inputs and counted 30,736 complete scored latents, expanded to 122,944 internal frames. The saved report records verified unchanged parameters and sequential stem-then-final calibration.
- The step 1,600 CPU checkpoint matches the saved calibration record. Both normalization layers remain frozen and their calibrated running buffers match hash ac9ec9191bbe629a28362b21605174478543b8bcecde1c284dd551030ae00e68.
- Prefix batches averaged 25.65 distinct sources and 22.83 languages, with at least 18 sources; at most 7 distinct nonoverlapping crops came from one source. There is no obvious single-language/single-utterance batch collapse.

## What remains weak

| Labeled event | First 1,000 updates | Entire plan |
|---|---:|---:|
| Crying |15.36s|140.99s|
| Giggling |15.03s|122.08s|
| Human whistling |9.85s|112.21s|
| Shouting |5.12s|41.03s|
| Yelling |0s|4.76s|

These are scored windows from files carrying event labels, not dense event annotations. The capacity-capped event share is approximately 0.56%, not the initially requested 5%. Rare examples are distributed through the run, so their early exposure is very small. This limits what current event scores can establish; it is not evidence that those windows were lost.

Calibration deliberately includes at least one example from many small groups. Its duration distribution therefore differs from optimization, with more other-language and expressive material. It covers 109 ordinary-speech languages plus Japanese JVNV mixed material, but no ordinary Japanese speech. This is a possible representativeness limitation for fixed normalization statistics, not a demonstrated cause of poor quality.

The heldout panel is 84 sources, 167 crops and 18 identified speech languages. Only 13 of 22 Indic languages are represented. Bodo, Dogri, Konkani, Kashmiri, Maithili, Manipuri, Odia, Sanskrit and Santali are absent. Chinese, Arabic and German are absent too. There are no dedicated crying, giggling, shouting, whispering or breathing condition groups. Whistling has only 2 sources/4 crops. These small groups cannot establish broad condition generalization.

At step 1,498, 13.48% of scored teacher sample-time lay in 20 ms regions below−60 dBFS; 60.06% was at least−40 dBFS. Teacher output had zero clipped samples and only 10 exact-zero samples. Quiet examples are present. The new waveform objective is raw valid-sample MAE without the old inverse-RMS amplification; legacy normalized values are detached diagnostics. Mel/log-mel still contribute to quiet audio, so this is not a claim of uniform gradient contribution.

## Recommended interpretation

Retain the existing panel unchanged for comparable learning curves. A separately reserved expanded panel is needed before any all-language or all-event quality claim. Do not infer a data-loss or calibration failure from the current weak rare-event scores. If those conditions remain weak, their seconds of actual scored exposure and the strict no-repeat constraint must be considered explicitly before changing the model or adding another loss.

This audit verifies saved identities, counters and calibration evidence; it does not independently rerun original audio hashes, decoder outputs, calibration inference or perceptual metrics. SourceCorpus performs audio hash/decoded-length checks when preparing targets. Unknown FLEURS speaker identities remain an acknowledged split limitation.

