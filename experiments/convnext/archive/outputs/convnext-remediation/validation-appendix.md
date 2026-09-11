# Reserved whispering and breathing validation appendix

Completed on the Runpod CPU with one thread. This is a separate panel; the original 84-source, 167-crop validation panel and its historical aggregates remain unchanged. No model, optimizer, teacher weights, training data or live training settings changed.

Seven existing reserved recordings provide 50.585 seconds of source audio: four labeled Whispering and three labeled Breathing. Continuous frozen teacher targets preserve each complete source's context. The fixed panel scores 12 distinct startup/tail crops, totaling 17.178 seconds. Sources shorter than 2.56 seconds have only the startup crop, so this is not an exhaustive evaluation of all source samples.

| Source label | Files / crops | Active correlation ↑, 1700 → 3000 | Normalized waveform error ↓ | Mel error ↓ |
|---|---:|---:|---:|---:|
| Whispering | 4 / 7 | 0.406 → 0.515 | 0.522 → 0.459 | 1.589 → 1.454 |
| Breathing | 3 / 5 | 0.384 → 0.431 | 0.591 → 0.569 | 1.848 → 1.836 |

Both conditions improve between the saved checkpoints. The 167 quiet windows still fail the current strict acceptance checks, and none of the active crops meets the final 0.99 waveform-correlation target. Neither checkpoint clips any samples in this appendix; the maximum student peak at step 3000 is 0.326. These are teacher reconstruction diagnostics, not PESQ, listening ratings or proof of perceptual quality.

## Integrity checks

- Exact file hashes, decoded sample counts, mono 16 kHz format and finite samples passed for all seven files.
- Their known source, hash, parent and person/session identities are disjoint from every source in all 320,512 optimization and calibration windows, including future training. The check covers 70,353 training/calibration source identities.
- The four distinct session identifiers are uploader groups. Speaker identities remain unknown. Known-identity checks cannot detect unknown shared speakers, re-encoded copies or acoustic duplicates.
- Each source was already reserved as development data by uploader. Six files carry CC-BY-3.0 metadata and one carries CC0 metadata. None enters training.
- The source labels come from the previous FSD50K inventory and were not independently confirmed by listening. The labels do not establish dense event boundaries or guarantee every selected crop contains the named event.
- The original frozen AudioVAE2 source and checkpoint were verified by SHA-256. Its weights exactly match this student's teacher identity. Targets use raw 64-channel posterior means and continuous 16 kHz input / 48 kHz output, without gain correction, clipping or fitted alignment.
- Both teacher and student remained unchanged. Both checkpoints used the identical saved appendix target tensors on CPU. The original panel's targets were prepared on CUDA; cross-backend target identity is not asserted across different panels.
- Full target preparation plus both evaluations use 17.72 MiB on Runpod. Only JSON metadata was copied locally; audio, targets and checkpoints remain there.

The same 12 crops can now be compared at subsequent retained checkpoints. Publish their metrics under a separate appendix label instead of merging them into the original panel's time series. The result exposes the weakness without requiring a random-weight restart.

## Evidence

- [Appendix identity](validation-appendix-identity.json)
- [Teacher target identity](validation-appendix-target-identity.json)
- [Step 1700 results](validation-appendix-summary.json)
- [Step 3000 and matched comparison](validation-appendix-step3000-summary.json)

Panel identity: `10a942d7eb667be7d3b8b39cfaf9c61b41acee7e5f1cf4b60d490daafcd42b34`.

Remote full evaluations are under `/workspace/fast-audiovae-convnext-20260909-r9/remediation/validation-appendix/`. The training plan, original validation panel and trained weights were not modified.
