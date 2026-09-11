# Quiet audio and peak audit

The current training mechanics pass the checkpoint audit. The evidence supports continuing from the trained weights, not restarting. Quiet residuals and transient overshoot remain separate unresolved quality problems; more training is helping, but cannot be promised to fix either completely.

This review reads the step 4,700 checkpoint, saved training history, the existing step 1,700/3,000 fixed-panel evaluations and the completed language/event appendices. The checkpoint/history audit performs no optimizer updates; the separate three-case synthetic CPU diagnostic below performs inference only. The 0.912 correlation is the separate new-language cohort, not a replacement for the original panel. At step 3,000 the original speech subset is at 0.900 and all active original-panel crops average 0.793.

## Scheduled step 5,000 update

The already scheduled original-panel evaluation finished during this audit. No extra full-panel evaluation was launched.

| Same original panel | Step 3,000 | Step 5,000 |
|---|---:|---:|
| Speech correlation | 0.9004 | **0.9302** |
| All active correlation | 0.7925 | **0.8505** |
| Mean quiet residual RMS | 0.0003383 | **0.0002733** |
| Median phase-template residual | 0.0001874 | **0.0001302** |
| Maximum absolute peak | 1.3483 | **1.2515** |
| Full-scale samples / affected crops | 112 / 3 | **348 / 5** |
| Quiet windows passing | 0 / 1,540 | **0 / 1,540** |

Quiet residual falls another 19.2% and the repeating residual 30.5%, so neither is flat yet. Peak excursions spread to Kannada and Sindhi speech as well as laughter. The five affected crops represent three source recordings; overlapping validation crops can count the same physical transient more than once. Maximum amplitude improves, but incidence does not. Step 3,000 was evaluated on CPU and this scheduled step 5,000 evaluation on CUDA, using unchanged target tensors, masks and source windows. This is not a new device-parity experiment.

## Actual encoded digital silence

A separate CPU, one-thread diagnostic filled the exact-input-silence gap using the real frozen encoder: one second of zeros, six seconds of zeros, and one second of zeros followed by one second of reserved speech. It did not train on these inputs or add them to the natural-speech aggregate.

All three produced the correct number of samples. Batch and one-latent-frame streaming outputs agree, with worst absolute difference **6.86e-7**. Teacher and student weights were unchanged. Encoded silence is not a zero latent vector: its latent RMS is approximately 0.65–0.68.

The six-second all-zero input leaves a steady student-minus-teacher residual around **0.0000992 RMS** in the final two seconds. This confirms persistent low-level error beyond startup. The teacher itself does not emit exact digital zero: its six-second RMS is approximately **0.0000378**, partly reflecting startup. Training should match that teacher response rather than blindly mute all quiet latents.

The diagnostic found no student-decoder batch/streaming accounting or boundary-state mismatch in these three CPU cases. It used batch teacher encoding, so it does not establish end-to-end encoder streaming parity. All 402 quiet windows across the three cases still fail the current checks; correct accounting is not quiet-quality acceptance. It strengthens the case for measuring the quiet floor separately from correlation, while the improving fixed-panel trend still supports keeping the current weights.

## Verified mechanics

| Area | Finding |
|---|---|
| Teacher and latents | The same frozen AudioVAE2 encoder supplies raw 64-channel posterior means to the student; frozen decoder waveforms remain the targets. The teacher executes with gradients disabled and checks its evaluation/frozen state. No target gain adjustment or per-clip normalization enters the reconstruction comparison. |
| Calibration | Completed at step 500 using 512 training-only windows and two identical passes. Both norms use 122,944 valid internal frames. Fixed parameters during calibration and frozen-buffer hashes still verify at step 4,700. |
| Training state | Strict checkpoint loading passes, metrics through step 4,700 match the checkpoint hash, the sampler cursor is 150,400, and the discriminator has exactly 4,200 updates. The balancer has 4,700 updates. Bound implementation hashes are unchanged. |
| Losses | Waveform, mel, adversarial and feature matching are active. The most recent 200 updates have no nonfinite values or balancer coefficient saturation. The optimizer remains Muon plus AdamW. |
| Valid samples | Reconstruction removes context and padded tails before scoring. Waveform reductions use actual sample counts; mel reductions use valid time-frequency counts. Scored masks prevent context/padding entering gradient norms. |
| Timing and context | Each original 40 ms latent produces 1,920 waveform samples at 48 kHz. Real causal history is supplied; right padding does not influence earlier causal output. Partial final latents retain only their valid waveform samples. |
| Perceptual alignment | Each example contributes an aligned student/teacher 9,120-sample segment. Every planned example is long enough; the engine rejects a silently dropped discriminator example. Teacher targets are detached, and generator backward must leave discriminator gradients empty. |
| Held-out data | Original targets and panel identities remain fixed. New event and language panels are reported separately and excluded from optimization/calibration. Four files per added language establish coverage, not a strong language ranking. |

The averaged measured output-gradient component proportions are now waveform **30.9%**, mel **40.1%**, feature matching **18.6%**, adversarial **10.3%**, close to the scheduled 30/40/20/10 split. These are norms before summation, not parameter-update shares. The earlier mel dominance has substantially resolved without a cap. There is no evidence here for adding that cap now.

Training waveform MAE averages 0.01114 over updates 2,801–3,000 and 0.00951 over 4,501–4,700; mel averages 1.201 and 1.131. These batches contain different audio, so their direction is reassuring but does not substitute for fixed-panel validation. Generator adversarial loss rises while discriminator loss falls as the discriminator learns; that alone is not a failure signal.

## Silence is supervised, but not solved

At step 3,000 all 1,540 quiet windows fail the current residual test. Of these, 1,002 also exceed the output-RMS limit. The remaining 538 fail waveform matching without excessive output RMS. It would be incorrect to call all failures loud hiss.

The thresholds are provisional engineering checks, not calibrated audibility limits: quiet means teacher RMS at most 0.001, residual must be below the larger of 14.14% of teacher RMS and 0.00001, and output RMS must remain within +1 dB of the teacher, with the same absolute floor. The original panel contains no exactly zero teacher windows. Exact-input digital silence was absent from the original natural panel and is now tested separately in the synthetic diagnostic above.

Nevertheless, the excess near silence is real. Among 149 windows with teacher RMS below 0.00001, mean residual is approximately 0.000198, or −74.1 dBFS. The median residual is 18.7 times its allowed limit. Audibility still depends on playback gain and surrounding sound.

There is ample low-level training exposure: the saved coverage contains about 13.7% of scored duration in 20 ms windows with teacher RMS below −60 dBFS and 3.44% below −80 dBFS. Adding expressive material is useful, but lack of all quiet audio is not the explanation.

Raw waveform MAE supplies a gradient on every erroneous quiet sample. The existing dedicated quiet and repeating-phase loss helpers are **not active in RecipeV2Engine**. This is a known objective choice, not an accidental zero gradient. Enabling the older quiet helper blindly would also reintroduce per-example/window weighting rather than the current valid-sample weighting, so it is not an approved quick toggle.

The repeating residual metric averages a 480-sample template and includes DC and phase-coherent error. It suggests an error related to the direct 480-sample output head, but does not prove that all noise is a 100 Hz tone or identify a defective convolution. Its median fell 30.2% between steps 1,700 and 3,000.

## Peak overshoot is a different problem

The full-scale excursions are loud interior transients, not silence. At step 3,000 three laughter crops contain 112 samples at or above full scale, with maximum absolute output 1.348. Their teacher peaks remain below one. The language appendix adds one Konkani crop with 16 such samples and a 1.137 peak. That crop has lower overall student RMS than teacher RMS, despite its local overshoot. A global gain change cannot resolve both errors.

The waveform output head is unbounded, consistent with its current architecture. General waveform and spectral losses penalize overshoot, but no dedicated peak penalty exists. Rare samples can contribute very little to an average loss, while the random 190 ms adversarial segment may miss a transient in a longer crop. This is a plausible limitation of the objective, not proof that the discriminator implementation is wrong.

Maximum peak improved from 1.398 to 1.348, but affected samples increased from 75 to 112. That mixed trend prevents claiming peak behavior is fixed. The metrics describe raw floating-point excursions, not proof that a saved playback file has already clipped them.

## Decision

Keep the current weights, calibration, optimizer and scheduled losses. The verified data continuation is running from the committed step 5,900 checkpoint and was confirmed progressing at step 5,982, with the observer at the same step. It adds 3.99 hours of unique expressive material while preserving every remaining original window. The scheduled step 5,000 results support continued reconstruction learning, while keeping peak incidence as an unresolved issue. Do not restart, normalize output volume, mute quiet audio or clamp the decoder to hide these failures.

If quiet residuals level off, the first targeted candidate should be a teacher-conditioned, valid-sample-weighted residual objective with a bounded gradient contribution, preserving real breaths and whispers. If peak excursions persist or spread, a separate training-only excess-peak penalty on the full valid crop is a reasonable candidate. Neither adds CPU inference work, but each needs an isolated comparison before promotion. They should not be introduced together with a data change and then credited without a control.

The new synthetic encoded-silence panel should remain distinct from real held-out speech and be repeated on the final acceptance checkpoint. Zero latent vectors are not a valid substitute for encoded silence.

Evidence: [checkpoint health](current-health-audit.json), [fixed-panel review](fixed-panel-step3000-summary.json), [language validation](language-validation.md), [event validation](validation-appendix.md).

Additional evidence: [scheduled step 5,000 results](fixed-panel-step5000-summary.json), [encoded-silence checks](encoded-silence-audit.json), [verified continuation status](continuation-live-status.json).
