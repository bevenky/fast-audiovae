# Expressive training data status

Verified on 8 September 2026. These completed sources provide **25.9147 training hours and 2.4638 development hours**, comprising 39,476 training utterances and 2,799 held-out utterances. Hours measure complete retained files, including their natural pauses. They do not measure time spent laughing, crying or making another specific sound.

| Source and pinned release | Train hours | Dev hours | License |
|---|---:|---:|---|
| [Thorsten emotional v02](https://www.openslr.org/resources/110/thorsten-emotional_v02.tgz) | 2.9343 | 0 | CC0-1.0 |
| [JNV ver3](https://ss-takashi.sakura.ne.jp/corpus/jnv/jnv_corpus_ver3.zip) | 0.2447 | 0.1502 | CC-BY-SA-4.0 |
| [JVNV ver1](https://ss-takashi.sakura.ne.jp/corpus/jvnv/jvnv_ver1.zip) | 2.9746 | 0.9695 | CC-BY-SA-4.0 |
| [CREMA-D, revision 1658cd3](https://github.com/CheyneyComputerScience/CREMA-D/tree/1658cd342dff90010aa843eaeebd53610a08b1dc) | 4.7113 | 0.5428 | ODbL-1.0 and DbCL-1.0 |
| [EmoGator, revision 51eeca5](https://github.com/fredbuhl/EmoGator/tree/51eeca515a95f168ab73391cd9e7c975ab964429) | 15.0498 | 0.8013 | Apache-2.0 |
| **Total** | **25.9147** | **2.4638** | Source obligations retained |

Original archives and MP3/WAV bytes are retained with SHA-256 receipts. CREMA-D audio also matches its pinned Git LFS hashes; EmoGator members match its pinned Git tree. Prepared audio is deterministic **mono 16 kHz IEEE FLOAT WAV**, carrying a separate file hash and original format metadata. Non-16 kHz files use whole-utterance SoXR VHQ conversion. JNV stereo files explicitly use channel 0 while preserving both original channels in the archive. No additional gain normalization or amplitude clipping is applied. Thorsten's publisher had already normalized its source recordings.

This preserves AudioVAE2's existing **16 kHz encoder input and 48 kHz teacher-decoder output**. The native 22.05/44.1/48 kHz recordings remain available for future work; the current real-audio reference supervises only the prepared speech band, with 8 kHz as an upper bound rather than a measured native bandwidth.

The splits keep held-out contributors separate. Thorsten's single speaker is training-only. JNV and JVNV conservatively group matching local speaker IDs across both corpora and hold M2 out. CREMA-D holds out actors whose IDs are divisible by ten. EmoGator selects twenty contributor IDs by fixed hash ordering; 341 contributor IDs remain across both splits after input exclusions. Known duplicate audio is excluded from the counted hours. This checks supplied identities and exact decoded duplicates; it does not establish acoustic deduplication across different encodings or unknown identities.

The metadata supports these specific coverage statements:

- **Whispering:** 299 explicitly labelled training clips, 21.90 minutes, one speaker.
- **General nonverbal vocalizations:** 314 JNV training utterances; 1,189 JVNV training utterances contain 21.60 minutes of publisher-marked generic nonverbal intervals. EmoGator adds 28,874 training vocal bursts across all thirty intended emotion categories.
- **Candidates needing action verification:** phonetic annotations identify 84 laugh-like, 26 sniffle/sob-like and 44 scream-like training utterances. Available JVNV timing annotations cover respectively 75.11, 34.07 and 18.80 seconds of nonverbal sound within those groups. These are phrase interpretations, not confirmed action labels.
- **Remaining gap:** separate verified giggling, crying, shouting, screaming and whistling coverage. Sadness is not counted as crying and anger is not counted as shouting. The additional event-labelled source acquisition remains a separate gate.

Integration checks passed. Every one of the 2,799 development WAV headers is mono 16 kHz FLOAT and matches its declared sample count. The shortest development files contain 24,024 samples in the four-source corpus and 11,520 in EmoGator, above the loss requirement of **1,366 input samples**. All training manifest lengths also exceed that minimum. Synthetic CPU fixtures using the default `SourceCorpus` reader verified both accepted preparation policies, exact reference samples, whole-utterance caching, 29 latent frames of context, masking of context and padding, and identical reference samples after reopening the disk cache.

The training CLI enables the explicit prepared-source policy. The original encoder and teacher decoder remain frozen in FP32; only the standalone student decoder enters the optimizer. Exact resume restores the optimizer, random state and no-replacement sampler, checking the manifest, teacher, preparation, implementation and runtime identities. The production launch must supply `--corpus-audit` to bind the completed main-corpus readiness report; that CLI option is optional for other uses. Corpus or implementation changes after launch require a new phase instead of an unchanged resume.

EmoGator excluded 1,706 non-mono files and twenty exact duplicates. Its diagnostics recorded 1,393 files with decoded samples at or above full scale before deduplication; MP3 overshoot means this flag alone does not prove source clipping. Those amplitudes were preserved. No retained file had more than 90% exactly zero samples. Quality and RTF evaluation remain deferred until 10,000 total training steps. This review used metadata, audio headers and synthetic CPU integration checks only.
