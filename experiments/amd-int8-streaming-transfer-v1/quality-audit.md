# AMD quality screen audit

The transferred optimizations preserve the existing selective INT8 decoder's waveform samples exactly on all six complete recordings. The existing quantization still has a small measured quality cost relative to FP32. These are separate findings.

This audit read the saved results and audio files only. It did not run a decoder or rescore the audio. Evidence: [quality scores](quality-scores.json), [local scoring manifest](quality-manifest.json), [export receipt](quality-002/result.json), [short timing receipt](timing-001/result.json), and [small stateful gates](gates-001/result.json).

## Scope and integrity

- **36/36 signals completed:** six original references plus six outputs from each of stock FP32, optimized FP32, existing selective INT8, transferred selective INT8 and Pocket continuous Mimi. There are zero metric errors, initialization errors or warnings. Each reconstruction has all 12 reported metrics; each original has the five no-reference predictions. Every primary metric has six finite values per applicable model.
- The six full recordings total **38.88 seconds**: Hindi, Bengali, Tamil, English, Latin American Spanish and French. There is one recording per language. This is a small speech screen, not a multilingual or expressive-audio qualification.
- All 36 local signal-file hashes match the manifest. The three scorer source hashes and all six metric-asset hashes match the recorded provenance and pinned manifests.
- Decoding used AMD CPU 0, one thread, ONNX Runtime **1.30.0**, CPUExecutionProvider only. Each decoder made **488 calls**. Packet counts, sample totals and the documented right trim reconcile for all 30 exports. No left trim, normalization, clipping, extra flush latent or graph switching occurred. Hindi and French end with a real 40 ms AudioVAE2 call using the same graph as the preceding packets.
- The four AudioVAE2 variants use identical frozen encoder latents and output at 48 kHz. Mimi uses its own encoder latents and outputs at 24 kHz. The sources are 16 kHz. Encoding was not rerun or timed; this is a causal streaming **decoder** comparison.
- Local predictor scoring used CPU only, one thread and the retained ORT **1.29.0** predictor environment. This is intentionally distinct from decoder ORT 1.30. UTMOS is **UTMOS22 strong via SpeechMOS**, not UTMOSv2. DNSMOS P.835 OVRL and P.808 are separate predictions.

## Mean source-referenced scores

All columns below are higher-is-better. The originals' reference-based metrics were not evaluated against themselves. Equal language weighting equals equal recording weighting for this six-clip cohort.

| Output | PESQ-WB | STOI | ESTOI | UTMOS22 | DNSMOS P.835 OVRL | DNSMOS P.808 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original audio | n/a | n/a | n/a | 2.14950 | 2.63499 | 3.35989 |
| Stock FP32 | 3.76850 | 0.933391 | 0.881048 | 2.09304 | 2.62107 | 3.40974 |
| Optimized FP32 | 3.76849 | 0.933391 | 0.881048 | 2.09304 | 2.62107 | 3.40974 |
| Existing selective INT8 | 3.72198 | 0.931639 | 0.877457 | 2.08707 | 2.58095 | 3.40404 |
| INT8 with transferred optimizations | 3.72198 | 0.931639 | 0.877457 | 2.08707 | 2.58095 | 3.40404 |
| Pocket Mimi | 2.17832 | 0.803916 | 0.638151 | 2.61151 | 2.81779 | 3.20949 |

Mimi leads UTMOS22 and P.835 even above the original recordings, while AudioVAE2 leads PESQ, STOI, ESTOI and P.808. A higher no-reference prediction is not proof of more faithful reconstruction. The two DNSMOS columns must not be merged or selectively substituted.

## Quantization differences by recording

Each entry is **existing INT8 minus optimized FP32**. The transferred candidate has the same differences. Negative numbers indicate a lower score.

| Language / UID | PESQ-WB | STOI | ESTOI | UTMOS22 | P.835 OVRL | P.808 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hindi `hi_in_00159_1941` | -0.02455 | -0.000577 | -0.001006 | -0.01079 | -0.05358 | -0.00676 |
| Bengali `bn_in_00582_1926` | -0.02150 | -0.001354 | -0.001713 | +0.00769 | -0.05997 | -0.00631 |
| Tamil `ta_in_00012_1740` | -0.01035 | -0.000784 | -0.001446 | -0.01409 | -0.05637 | -0.01780 |
| English `en_us_00287_1786` | **-0.15330** | **-0.006986** | **-0.015688** | -0.00013 | -0.01651 | -0.04097 |
| Spanish `es_419_00706_1731` | -0.02749 | -0.000254 | -0.000980 | -0.02745 | -0.02804 | -0.01329 |
| French `fr_fr_00045_1736` | -0.04187 | -0.000562 | -0.000715 | +0.00895 | -0.02625 | +0.05093 |
| **Mean** | **-0.04651** | **-0.001753** | **-0.003591** | **-0.00597** | **-0.04012** | **-0.00570** |

All six recordings lose some PESQ, STOI, ESTOI and P.835 score under the existing quantization. Four lose UTMOS and five lose P.808. The English recording has the largest reference-based loss and is also the quietest original, RMS **0.000602**. This association identifies a useful listening example; one clip cannot establish quietness as the cause. The native INT8-versus-stock waveform SNR spans **26.64–39.06 dB**. None of the 30 reconstructed outputs exceeds full scale; the largest absolute sample is **0.61158**.

## What passed numerical checks

For each of the six complete utterances, the transferred and existing INT8 outputs have identical decoded float32 samples. This was independently verified with `soundfile.read` on the six local scoring pairs. The raw, untrimmed PCM hashes in the export receipt also match for all six. Raw WAV files remain on the remote machine; this audit did not download them.

The **WAV container hashes differ** because their `PEAK` metadata differs. That does not change the audio samples. Say “bitwise-identical waveform samples,” not “identical WAV files.” All listed PESQ, STOI, UTMOS and DNSMOS scores for the two INT8 arms are exactly equal. ESTOI differs by at most **2.78e-15**, consistent with numerical noise in the scorer on identical samples.

The full-utterance receipt contains six unique transfer waveform comparisons and **108 unique final-state comparisons**. Every one is bitwise exact. The same comparisons are logged twice by the immediate and final group checks, yielding **228 records**; they are not 228 independent recordings or windows. The receipt has 240 passing check records in total, including the duplicated FP32-versus-stock comparisons. The optimized FP32 waveform maximum absolute difference from stock is **4.78e-7**, within the established tolerance. Their state layouts differ, so their state values are not incorrectly compared or exchanged.

The separate short test retains **836/836 bitwise checks** for zero/small/real latents, odd packet partitions, reset reproducibility and future-mutation prefixes. The timing screen has **285/285 bitwise transfer comparisons**. These establish the stated narrow implementation parity, not universal perceptual transparency of INT8 against FP32.

## Interpretation limits

The scorer resamples to a common 16 kHz band without external gain fitting or alignment. These scores do not evaluate AudioVAE2's 8–24 kHz output band. All six references are shorter than the DNSMOS 9.01-second window; all 36 signals use the pinned official waveform-repetition rule, and every repeat flag is present.

Do not use the scorer's bootstrap intervals as confidence evidence here. Its within-language bootstrap has only one recording in each stratum, so the intervals collapse to the point estimate. Report the six paired differences and their mean instead. The gain supported by this screen is faster execution with no added INT8 waveform change; the existing FP32-to-INT8 quality tradeoff remains measurable.
