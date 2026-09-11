# What the refinement screen missed

The correct base is the saved **step-8,090 student**. Every completed branch restored that model and its original Muon/AdamW state before applying its change. No student was trained from scratch in this screen. The output filter alone added seven new student parameters, initialized as an identity filter. Tanh has no trainable parameters. Fresh complex/magnitude spectral discriminators are training-only modules; their random initialization did not replace the trained student. All existing student weights remained trainable, while the AudioVAE2 encoder and teacher decoder remained frozen.

A new layer cannot be assumed to behave like the teacher merely because it is present in its architecture. Identity initialization protects the starting function; adaptation still needs a useful gradient and relevant examples. In this case the experiment did not adequately control those two requirements for every candidate.

## The important experimental gap

We copied historical gradient-normalization statistics when replacing the discriminator. Those old statistics came from a trained discriminator with very different gradient magnitudes. The nominal loss settings therefore did not describe the contributions the student actually received.

| Branch | Waveform | Mel | Feature matching | Adversarial |
| --- | ---: | ---: | ---: | ---: |
| Intended | 30.00% | 40.00% | 20.00% | 10.00% |
| Unchanged control | 30.57% | 39.27% | 20.22% | 9.94% |
| Short mel | 34.13% | 31.50% | 22.93% | 11.45% |
| Fresh magnitude discriminator | 38.69% | 50.29% | 10.12% | 0.90% |
| Complex discriminator | 43.17% | 56.56% | 0.148% | 0.122% |

These percentages are mean per-batch scaled output-gradient norms divided by their sum, before vector addition. They are not parameter-update shares or causal attributions to quality.

The complex branch received only **0.27% combined perceptual contribution**, versus the intended 30%. Even its final 20 updates averaged only 0.44%. The fresh magnitude control averaged 11.02%, so the discriminator pair did not have comparable effective objective balance. The EMA decay of 0.999 retains about 82% historical influence after 200 updates. More iterations would eventually adapt it, but silently waiting for that is a poor experimental control.

The historical-norm mechanism follows the [EnCodec balancer implementation](https://github.com/facebookresearch/encodec/blob/main/encodec/balancer.py). The stale-state finding and percentages above come from our own frozen source and all 1,200 saved generator updates, recorded in [gradient-audit.json](gradient-audit.json).

The repair is to preserve model weights, optimizer moments and calibrated model normalization, while explicitly recalibrating gradient statistics for the losses whose definitions or discriminators changed. These are different forms of state. A fixed-weight calibration should establish the new gradient distributions before generator updates resume. Gradually enable the corrected contribution if necessary and verify it rather than relying on a nominal setting.

Short-window mel had an additional ambiguity: the original scales went from weight 1/3 each to 1/5 each when two resolutions were added. Combined with inherited normalization, its achieved mel contribution fell from 39.27% to 31.50%. Its result tests a changed scale mixture and changed effective influence, not shorter windows alone. A corrected trial must declare the internal weights and match the overall spectral influence.

This changes the interpretation of the experiment. The saved quality numbers are valid for the models actually trained. They do not establish that complex discrimination or short-time supervision lacks value.

## Verification of the actual audio subset

The generator branches each consumed **4.096 hours, 6,400 nonoverlapping scored windows, from 1,714 sources**. All 22 scheduled Indic languages are represented. The manifests contain 110 known language labels; labels are not a claim that every label is an independent language or that every language has enough examples for a strong ranking.

By duration, the main sources are FLEURS 58.81%, LibriSpeech 26.19% and IndicVoices 7.50%, with the remaining material from the expressive collections. Broad expressive labeling reaches the nominal 5.00%, but **4.42% is generic emotional/nonverbal material and only 0.59% is explicitly labeled material in the intended edge-case mix**.

| Named training condition | Scored crops | Audio duration |
| --- | ---: | ---: |
| Human whistling, source-description label | 1 | 1.12 s |
| Screaming | 5 | 10.99 s |
| Laughter | 17 | 32.96 s |
| Crying/sobbing | 2 | 5.12 s |
| Giggling | 2 | 3.19 s |
| Breathing | 2 | 2.10 s |
| Whispering | 1 | 0.54 s |
| Explicit whisper-style speech | 12 | 25.43 s |

These counts follow source labels and descriptions; they do not establish that the named event occupies every scored crop. Event presence within the actual selected crop still needs qualification. The held-out panel covers all 22 Indic languages but lacks distinct crying, giggle, shout and yell groups. This is enough breadth for a general adaptation screen but insufficient coverage to judge learning on every named failure. A generic emotion label must not stand in for verified whistling, crying or breath. The next subset needs explicit duration quotas for these categories, with multiple sources, while retaining multilingual ordinary speech and no repeated scored intervals within a branch.

**The model did see natural silence and low-level input.** Direct source-PCM inspection found 41.51 minutes of complete 20 ms windows with input RMS at most 0.001, or 16.90% of the measured input windows. It also found 82.78 seconds of exactly zero input windows. These are input measurements, not the teacher's 48 kHz output-target statistics. The absence of a dedicated synthetic-silence fixture from training does not mean the model never saw silence, and data absence alone cannot explain the persistent quiet floor.

All 2,021 selected/held-out files passed byte hashes, mono 16 kHz shape/sample-count checks and finite-PCM checks. Recorded source/audio/parent-time intervals did not overlap within an arm, and training remained disjoint from held-out identities and 11,317 reserved rows. No official dev/test partition was selected for training. Known speaker/session identifiers also had no intersections; unavailable speaker identities are not proof of universal speaker disjointness. All six saved panels and target identities matched. [Full data and initialization evidence](data-audit.json).

## What the architecture evidence actually says

**Tanh addresses overshoot.** At an amplitude of 0.0002 its derivative is about 0.99999996, so it cannot materially attenuate the quiet floor. Its bounded output worked, but further adaptation is required to recover the measured reconstruction tradeoffs.

**The seven-tap filter learned almost identity.** Its coefficients sum to 1.0031763, with gains of approximately +0.0275 dB at DC/100 Hz and -0.017 to -0.028 dB at 8–24 kHz. It is not meaningfully suppressing those components. Its tiny direct gain cannot alone explain the larger quiet regression, because the entire student also changed during fine-tuning.

**Startup padding cannot directly fix a mature residual.** The valid-length comparison showed identical mature output. All 36 checks on trained streaming models passed, including empty calls and exact sample counts. The worst batch/stream discrepancy was 9.24e-7; encoded-zero discrepancies were much smaller than its approximately 2e-4 residual.

**The periodicity measurement is incomplete.** A 480-sample template corresponds to 10 ms at 48 kHz and contains DC plus harmonics at 100 Hz spacing. It does not prove a 100 Hz tone. The adapter has four phases per original latent and may also produce a 1,920-sample/40 ms pattern. The next diagnostic should separate DC, 480-phase and 1,920-phase components and genuine boundary jumps before choosing a remedy.

**Quiet input is not a zero latent vector.** The teacher's actual encoded-silence latents are nonzero. Its saved steady output RMS is about 9.63e-6, while the control student's steady RMS is about 22 times higher. The relevant constraint is reproducing the teacher response to those actual features. The existing final projection already has no bias; simply removing a bias or testing zero latents would miss the problem.

## A teacher-anchored next comparison

1. Start from the same step-8,090 checkpoint. Preserve the learned student and optimizer moments. Use a separate, versioned calibration for changed loss-gradient statistics.
2. Build a small training-only diagnostic pool with multilingual ordinary speech, low-level speech, real quiet intervals, whisper/breath, named nonverbals, and transitions into and out of silence. Verify durations per category rather than counting a broad emotional label as every desired event. Keep the evaluation sources reserved.
3. Measure existing loss gradients with weights fixed: achieved component shares, quiet versus active windows, short versus long spectral resolutions, and whether their directions oppose one another. Log actual optimizer update/weight ratios before changing clipping. All current updates clip, but this does not establish a proportional slowdown under Muon/AdamW.
4. Use the frozen teacher as the reference on the same latents and histories. Separate student-to-teacher reconstruction from teacher-to-original quality. Also establish the reference implementation's numerical repeatability before turning tiny residual thresholds into an acceptance criterion. Do not force the teacher's room tone, breaths or low-level speech to zero.
5. Run a corrected matched discriminator comparison only after its measured contributions are useful. Judge new discriminators after adequate adaptation with sparse common-panel checks. Keep the short-window weighting question separate. A longer bounded screen may be appropriate, but no duration guarantees quality.
6. If quiet error remains, use the measured feature/gradient evidence to choose a targeted correction in the existing head before adding a new layer. A head-only diagnostic would test whether the current features can express the teacher's quiet response while preserving speech. It is a hypothesis to test, not an established fix. An earlier quiet-phase penalty already failed under a different recipe, so repeating it blindly is not the next step.

These calibration, data and training-objective changes add no CPU decoder work. The original run remains paused, and no corrected training run has been launched.

