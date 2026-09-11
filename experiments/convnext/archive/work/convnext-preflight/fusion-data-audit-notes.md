# Completed paired screen: exact data audit

Report: `outputs/convnext-fusion-experiments/data-audit.json`

The audit read the frozen experiment identity, preserved parent receipt, sealed continuation plan, six completed arm reports and their previously tensor-verified summary. It rehashed 2,021 selected and held-out audio files, about 904 MB, and checked the prepared PCM shape, length and finiteness. No model inference, benchmarks or training were run.

All six arms started with the same trained step-8090 student and its existing optimizer state and ended at step 8290. The filter's seven causal taps started as `[0,0,0,0,0,0,1]`. It was an identity addition to the trained model, not a randomly initialized replacement decoder. All arms saw the same 6,400 generator crops. Only the fresh-magnitude and complex discriminator arms additionally saw the same 640 distinct crops for discriminator-only warmup.

The 6,400 generator crops contain 4.096 hours from 1,714 sources, covering 110 known normalized language codes, including all 22 scheduled Indic languages. English contributes 29.96% of scored duration. Source starts occur in 1,624 crops, 25.375%, so startup is not absent from this screen.

The nominal 5.004% expressive allocation is 4.417% generic Emogator emotional nonverbals and only 0.587% explicit source-labelled actions. Exact named-action crop exposure is thin:

| Label | Crops | Sources | Scored seconds |
| --- | ---: | ---: | ---: |
| Laughter | 17 | 12 | 32.96 |
| Explicit whisper style | 12 | 7 | 25.43 |
| Screaming | 5 | 3 | 10.99 |
| Crying and sobbing | 2 | 1 | 5.12 |
| Yell | 3 | 2 | 5.07 |
| Giggle | 2 | 2 | 3.19 |
| Breathing | 2 | 2 | 2.10 |
| Human whistling | 1 | 1 | 1.12 |
| FSD Whispering | 1 | 1 | 0.54 |

These are durations of selected crops from recordings carrying those source labels. They are not manually verified event occupancy inside every crop. No Shout or Chuckle_and_chortle crops occur in this 200-update generator slice. Generic emotion cannot be counted as evidence of a particular laugh, cry or whistle.

There are no dedicated synthetic silence, noise or fade training fixtures. However, it would be wrong to say the student never saw silence. The selected natural input PCM contains 41.51 minutes of complete 20 ms windows with RMS at most 0.001, 16.90% of inspected windows. It contains 82.78 seconds of exactly zero 20 ms windows and 127.36 seconds at RMS at most 0.00001. These are input measurements, not frozen teacher-output quiet targets. They do not prove the model saw long contiguous encoded-silence trajectories.

The fixed development panel contains 281 natural crops from 143 sources plus three six-second synthetic fixtures, with all 22 Indic languages represented. It has held-out laughter, screaming, whispering, breathing and whistling. It lacks separate named crying, giggle, shout and yell groups, so those behaviours cannot be individually validated from this panel. Quiet diagnostics use 3,336 natural teacher-quiet windows. Overlapping crops are not independent examples or unique acoustic events.

No source, file-hash, parent-recording or known speaker/session identities intersect training and the held-out panel. All 11,317 reserved rows were checked. No official dev/test partition enters training. Selected scored intervals have zero overlap by source, file hash or absolute parent timeline, including discriminator warmup, inherited parent exposure and calibration. Shared causal context is intentional. Matching data across separate debugging arms is also intentional.

The checks depend on available identities and exact file bytes. They cannot detect an undocumented speaker alias or a re-encoded duplicate without an acoustic duplicate audit. The panel is development data repeatedly used for model selection, not an untouched final test set.

The existing source collection and baseline are suitable for a controlled screen, but this short generic slice is weak evidence for named edge-case adaptation. A further authorized edge-case screen should keep a common speech control and reserve enough distinct named events to measure changes. It should not claim missing silence as the sole explanation or infer a fix from these coverage counts alone.
