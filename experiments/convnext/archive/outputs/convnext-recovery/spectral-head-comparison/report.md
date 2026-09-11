# Teacher spectral supervision and head adaptation: results

Full-head adaptation with teacher spectral supervision is the strongest candidate from this bounded comparison. It substantially improves stationary silence and spectral fidelity without adding inference operations. It still fails the predeclared acceptance checks, so no candidate replaced the original step 8,890 checkpoint.

All four cases used the same 2,048 distinct training recordings, the same order, 256 updates of eight recordings and a separate 256-recording selection split. The frozen encoder supplies the same AudioVAE2 64-channel latent means. The latent adapter, normalization and ten ConvNeXt blocks stayed frozen. Joint adaptation updates only the existing head convolution, shared PReLU and output projection; projection adaptation updates only the last projection.

The waveform objective was the preceding selective silence objective: match teacher quiet waveforms while penalizing changes to the original student's nonquiet output. The spectral arms additionally matched teacher linear and log mel magnitudes using the existing three-resolution reconstruction implementation. No GAN, new activation, output clamp or synthetic silence fit was added.

## Matched final results

Percentages below compare with the unchanged original checkpoint. Reductions mean lower error; the one positive mel change is a regression. Canonical results use the same 285 crops and 147 sources, including 144 natural sources.

| Updated layers and supervision | Stationary silence error reduction | Natural quiet waveform error reduction | Average mel-error change | Natural sources exceeding 1% mel regression |
| --- | ---: | ---: | ---: | ---: |
| Projection, waveform only | 9.2% | 1.3% | −0.07% | 2 |
| Full head, waveform only | 25.3% | 3.4% | +0.36% | 25 |
| Projection, waveform + teacher spectral | 11.7% | 0.7% | −1.76% | 2 |
| Full head, waveform + teacher spectral | **49.2%** | **2.0%** | **−2.79%** | **2** |

The full-head waveform control reproduced the previous experiment's four trained head tensors exactly. That helps separate the effect of spectral supervision from an accidental change in the baseline or optimizer execution.

Full-head spectral adaptation improved mel error on 141 of 144 natural recordings. The earlier failing Cantonese source now improves rather than worsening: its waveform MAE falls 2.64%, and spectral error improves in quiet, active and transition windows. The final candidate also reduces quiet waveform error on all 87 natural sources with eligible quiet samples.

The spectral partition uses intact FFT windows classified from fixed teacher masks. Synthetic fixtures are excluded from this table, and every arm uses identical window counts.

| Natural-audio spectral error change | Full head, waveform only | Full head, waveform + teacher spectral |
| --- | ---: | ---: |
| Entirely quiet FFT windows | +1.85% | **−15.55%** |
| Entirely active FFT windows | +0.20% | **−1.98%** |
| Windows spanning quiet/active transitions | +1.86% | **−3.79%** |

These results support two conclusions. Explicit teacher spectral supervision addresses a real weakness in the waveform-only repair objective. Jointly adapting the existing nonlinear head gives a better result than changing only the final projection under the same spectral objective and update budget. This does not prove the head lacks capacity, that new layers are needed, or that every optimization schedule would produce the same ranking.

## Why the leading candidate is not accepted yet

Natural quiet waveform residual RMS improves only 2.02%, below the required 10%. It moves from 0.000258495 to 0.000253262. Stationary residual RMS moves from 0.000043793 to 0.000022232, but all 200 stationary windows still fail the existing strict quiet check. Natural failures decline from 3,386 to 3,344 of 3,434 windows. This is progress, not teacher equivalence.

Two natural recordings exceed the new source-level spectral guard:

| Recording | Original per-source mel change | Pooled spectral change | Location of the remaining regression |
| --- | ---: | ---: | --- |
| Bodo: `3659174697300869_chunk_1.flac` | +1.24% | +1.56% | Active speech; no eligible quiet FFT windows |
| Manipuri: `6473924464473013_chunk_1.flac` | +1.25% | +1.58% | Mainly active speech |

The same two sources regress under projection-only spectral adaptation. That suggests a shared supervision or optimization tradeoff worth investigating; it does not identify a new architectural defect. These held-out recordings should remain evaluation data rather than become targeted fitting examples.

Local quiet spectra reveal another limitation hidden by whole-recording averages. Five of the 79 natural sources with eligible quiet FFT windows worsen by more than 1% in that region. The largest is a screaming recording, `freesound:220655`: quiet mel rises from 0.687479 to 0.769676, or 11.96%, despite better overall mel. Its support is sparse: eight 1,024-point frames and two 2,048-point frames pooled across two overlapping crops, with no eligible 4,096-point frames. A separate Manipuri source worsens 4.86% on just four short frames; `freesound:220663` worsens 4.32%. These sparse local results need careful interpretation, but they prevent claiming that quiet spectral regressions are resolved. Waveform quiet coverage includes 87 natural sources; requiring intact quiet FFT windows reduces spectral coverage to 79.

Peaks are also unresolved. The overall maximum falls from 1.296632 to 1.292609, but laughter peaks rise from about 1.199306 to 1.199867. Scored overshoot observations increase from 588 to 589, including wider overshoot in overlapping Sindhi crops. These counts are observations on the unchanged panel, not deduplicated physical events. The candidate fails the no-new-peak-regression guard despite its lower global maximum.

Overall waveform MAE rises 0.084%, while mel improves 2.79%. This small aggregate waveform change passes the existing screen; it is not a claim of unchanged perceptual quality. No new listening or full MOS campaign was performed for these unqualified candidates. All candidates also failed selection on the separate 256-source split; the leading candidate had five source spectral failures there at step 256.

## Recommended direction

Retain full-head adaptation plus teacher spectral reconstruction as the leading recipe for the next investigation. Keep AudioVAE2's exact latent interface and the inexpensive causal ConvNeXt/direct-waveform decoder structure. The current experiment establishes useful gains with existing layers, so it does not yet justify adding channelwise PReLU, a wider head or more ConvNeXt blocks.

Before a longer run, examine the remaining active-speech spectral tradeoff using similar training-only examples and the fixed per-source guards. Preserve the full teacher target as the eventual objective everywhere; anchoring nonquiet output to the old student is a temporary safeguard, not a permanent replacement for teacher reconstruction. Keep peak bounding as a separate jointly adapted experiment, as previously planned. None of those follow-up runs has started.

Retain the local quiet and transition breakdown alongside whole-source scores, including support counts, so average improvements cannot conceal brief regressions.

## Calibration, integrity and runtime

The first four fixed training batches supplied a common spectral coefficient of 8.0578988, calibrated from waveform-output gradient energies. This was frozen across both layer scopes. It establishes equal initial gradient energy over those batches, not a constant balance throughout training. The first-batch finite-update check accepted learning rate 0.000001 for all four arms. AdamW states were fresh and matched, with no weight decay and gradient norm cap 1.

All 27 dedicated checks passed locally and on the Runpod runtime: PyTorch 2.14.0, CUDA 12.6 and cuDNN 9.25.1. Cached head replay was exact for every training and selection recording. No selected crop was too short for spectral supervision. All canonical output counts and sealed masks remained unchanged: 26,206,830 scored samples. Original engine state, checkpoint, frozen parameters, normalization and teacher-pair caches passed preservation checks.

The bounded fitting and evaluation phase took 129.3 seconds after initial context loading. An SSH interruption delayed result retrieval, but all four runs completed and their preservation reports were recovered. This is H100 experiment duration, not CPU RTF. No candidate passed quality, so no candidate CPU timing or deployment promotion followed. The inference graph has no added operations in any arm.

Full reports and experimental head tensors remain on the pod under `/tmp/fast-audiovae-recovery-20260909/spectral-head-comparison-v1`. Local [results.json](results.json) contains identities, calibration, validation summaries, canonical source/region metrics, preservation evidence and remote artifact hashes. [summary.json](summary.json) contains compact comparisons. The original checkpoint remains the retained model; no changes were committed or pushed.
