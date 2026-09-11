# Data, sampling and validation audit at step 10,000

The new phase consumed unique audio correctly, but its exposure was heavily concentrated in FLEURS. The saved validation loss does not establish quality across the training languages. These are concrete design gaps; neither proves that the decoder failed to learn or explains a numerical loss floor on its own.

This audit used the completed Runpod checkpoint, source manifests, readiness report and matching sampler implementation on 9 September 2026. No audio was read or inferred, and no training or quality benchmark was run. Exact metadata and reconstructed exposure are in [data-exposure-10000.json](data-exposure-10000.json).

## What actually ran

The assembled corpus contains 510.9274 train hours, 183,802 train utterances and 2,839 dev utterances. All 102 FLEURS configurations and all 22 scheduled Indian languages passed the acquisition gate. The sampler has 111 normalized labels, including `und`; that is not a claim of 111 identified spoken languages.

The run imported the original step-1,000 student and optimizer, then made 9,000 new updates with batch 64. These used exactly 576,000 nonoverlapping scored segments, totaling **371.6965 hours**. Qualifying short tails explain why this is below the nominal 409.6 hours of 576,000 full 2.56-second windows. Another **139.1593 usable hours / 221,743 segments remained**. Downloading 511 hours did not mean all 511 hours were used.

The source manifest matches its saved readiness hash; that readiness matches the checkpoint. The sampler implementation hash and identity match. Independently reconstructing all 576,000 selections reproduced the saved cursors and next-language position exactly. There is no evidence of window loss or repeat in this new phase. The earlier 1,000-step, 8-example bootstrap used sampling with replacement, so the no-repeat claim should not be applied retroactively to the entire 10,000-step history.

## Exposure imbalance

| Source | Available hours | Scored hours used |
|---|---:|---:|
| FLEURS | 340.20 | 322.35 |
| IndicVoices | 44.04 | 40.61 |
| LibriSpeech | 100.00 | 1.99 |
| EmoGator | 15.05 | 2.60 |
| CREMA-D | 4.71 | 0.40 |
| JNV + JVNV | 3.22 | 1.85 |
| Thorsten emotional | 2.93 | 1.13 |
| FSD vocal + human whistling | 0.77 | 0.77 |
| **Total** | **510.93** | **371.70** |

The sampler cycles languages equally while they have data, interleaves datasets by utterance, and consumes each utterance's windows in order. It does not implement weighted English/Indic/expressive exposure or explicit speaker balancing.

Consequently, FLEURS supplied **86.72%** of scored audio. All English sources together supplied only **4.01 hours, or 1.08%**; all expressive sources together supplied 6.75 hours. Every normalized label received data, but large English and expressive allocations were substantially underused. Known-speaker coverage was broad where IDs exist: 222 LibriSpeech readers and 9,381 IndicVoices speakers were touched. FLEURS has no verified speaker IDs, so its speaker diversity cannot be counted.

The training distribution also changed over time as smaller language queues emptied: active labels fell from 111 after the first 1,000 new updates to 17 at completion. Different loss windows therefore do not evaluate a constant input distribution. Most exhaustion happened late, with 98 labels still active at step 9,000. This cannot by itself explain the earlier plateau and does not establish that data order caused one.

## Validation is unrepresentative

| Dev source | Utterances |
|---|---:|
| EmoGator | 1,530 |
| CREMA-D | 738 |
| JNV + JVNV | 531 |
| LibriSpeech | 25 |
| FSD vocal + human whistling | 15 |
| FLEURS / IndicVoices | **0 / 0** |

The final reported loss, **25.1725**, averages utterances equally. EmoGator alone carries 53.89% of that mean. Dev language labels are English 763, Japanese 531 and unspecified 1,545, with no held-out Indic or broad multilingual speech coverage.

Evaluation scores only the first up-to-2.56 seconds of each dev utterance: 1.6978 scored hours from 2.5179 full dev hours. It does not assess interior stream continuation or chunk seams. It reports no per-source/language breakdown, original-teacher loss on this same panel, or student-at-step-1,000 baseline on this expanded dev set. Therefore the older 25-clip English warmup dev loss and the final expanded dev loss are not directly comparable. Final-only validation followed the user's instruction, but it leaves no comparable held-out learning curve.

## Supervision and conclusion

Every training input and real-reference target is mono 16 kHz. Some originals were higher-rate, but their prepared inputs and references are 16 kHz. The checkpoint pins whole-utterance FP32 raw-mean latents and 48 kHz original-teacher output. Frequencies above the input band are supervised by the teacher, without a native highband recording target in this phase. This is consistent with the retained AudioVAE2 interface; it is a limitation on what the experiment establishes.

No metadata evidence here identifies corruption, leakage or a broken sampler. The recorded checks cover canonical identities and known file hashes, not acoustic duplicate detection or recording quality. Before deciding that the architecture or optimizer failed, compare student and original teacher on the same representative held-out panel, separate source/condition scores, and compare loss components against the teacher's own reference mismatch. Any subsequent training should declare its desired exposure weights and unused-window allocation explicitly. The current loss number alone cannot establish audible failure or teacher-quality parity.
