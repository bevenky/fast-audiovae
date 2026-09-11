The natural-source diagnostics support keeping 30 frames of decoder history. They also expose a remaining student output floor and a separate historical encoder-cache discrepancy that should be audited before training resumes.

The [natural panel](natural-history.json) contains eight distinct source identities, two each for speech, laughter, whistle and naturally quiet intervals. Each comparison scores the same 640 ms of real audio with 48–143 available preceding latent frames. Seven cases have authenticated raw recordings for amplitude tests, covering all four groups; the second quiet case, from the Yell recording, has no raw-manifest entry in this repair. Its cached-history tests remain valid. The earlier encoded-control and Yell-only panels are separate evidence, not substitutes for these natural cases.

30 frames are sufficient on these examples. At 0 frames, reset errors reach 0.188 in the teacher and 0.260 in the students. At 8 frames, noticeable reset dependence remains. By 29 and 30 frames, it is at FP32 numerical scale; adding 40 frames does not remove the remaining student-to-teacher error.

| Model | Mean reset-versus-longest RMS at 30 frames | Maximum sample difference | Mean student-versus-teacher RMS with longest history |
|---|---:|---:|---:|
| Teacher | 2.74e-8 | 1.51e-6 | n/a |
| Parent, step 8090 | 3.02e-8 | 7.45e-7 | 0.006718 |
| Targeted, step 8490 | 2.94e-8 | 6.71e-7 | 0.006583 |
| Complex, step 8490 | 2.97e-8 | 8.34e-7 | 0.006669 |

These are equal-length diagnostic windows, not a new full quality benchmark or an analytical receptive-field proof. All 32 future-latent perturbation checks, teacher plus three students across eight cases, preserve the earlier output prefix bitwise. Teacher repeatability is bitwise on seven cases; the remaining repeated full decode differs by at most 7.15e-7. There is no evidence here that insufficient decoder history explains the persistent reconstruction error.

A historical encoder-cache discrepancy needs a separate audit. For the two laughter sources, freshly encoding the authenticated original input does not reproduce the cached latent sequence. The [bounded follow-up](natural-history-alignment-audit.json) rules out a simple 64-frame truncation or right-tail explanation: every one of the first 64 frames differs, and encoding only the first 64 frames gives essentially the same fresh latents as encoding the full recording.

| Laughter source | Fresh full-source versus cached latent max difference | Fresh full versus truncated-input latent max difference | Cached latent decode versus its cached target max difference |
|---|---:|---:|---:|
| 119459 | 1.035 | 1.23e-5 | 3.99e-7 |
| 386521 | 1.366 | 0 | 0 |

The [read-only cache audit](laughter-cache-provenance.json) shows that both original cache files match their saved checksums, and their latent prefixes match the heldout tensors exactly. Their stored original PCM is bitwise identical to the newly decoded source files, including the recorded prepared-PCM hash. Their metadata declares the same source/checkpoint, FP32 posterior means, disabled autocast/TF32, deterministic execution and disabled noise. Thus source-file drift, a mismatch in the stored prepared PCM and this truncation hypothesis are ruled out by the checks. The other five authenticated cases reproduce cached latents within max1.67e-5 and scored teacher audio within max1.34e-6. The precise historical encoder execution cause in the two laughter cases remains unresolved. The cached latent-to-teacher-waveform pairing is internally consistent; this is not evidence that all cached pairs are corrupted. Keep the fresh laughter amplitude baseline separate from the cached-latent baseline, and audit original encoder/cache generation before a training restart.

The amplitude experiments are internally matched: every factor, including 1, is independently re-encoded from the same complete authenticated source recording. The student and teacher always receive that same new latent sequence. At factor 0, all seven source-duration variants converge to nearly the same model-specific output RMS:

| Model | Output RMS on re-encoded zero input | RMS level relative to full scale |
|---|---:|---:|
| Teacher | 9.63e-6 | -100.33 dB |
| Parent | 2.16e-4 | -73.33 dB |
| Targeted | 1.43e-4 | -76.87 dB |
| Complex | 1.23e-4 | -78.22 dB |

The continuations reduce the student's floor, but it remains roughly 22–27 dB above the teacher across the three checkpoints. These are finite output levels, not numerical failure or an audibility judgment. The quiet natural window from source 25794 has teacher RMS9.67e-6 and student residual RMS2.19e-4, 1.37e-4 and 1.22e-4 respectively. Low-amplitude inputs also scale less proportionally in several students than in the teacher; for laughter386521, fitted log-RMS slopes are 0.543/0.562/0.604 versus teacher0.964. Those are response secants over factors1 to0.01, not local Jacobians.

Phase dependence and alignment do not provide a universal fix. On the two quiet windows, a 1920-sample phase template explains approximately 50–98% of residual power across checkpoints; the 480-sample results are similar. This supports investigating a repeatable structured output floor, while leaving its architectural cause unproven. On active speech and laughter, a diagnostic lag search of ±48 samples plus gain0.8–1.25 usually removes only 0–5% of squared error. One complex-model whistle example improves by56% with lag-6 and gain1.154, so alignment matters for that case. Quiet fits reach the gain lower bound and some lag limits; they do not establish a global delay correction. Raw waveforms and primary scores were never shifted or rescaled.

The practical priorities from this panel are to retain the current 30-frame history, resolve the two historical encoder-cache identities, and target the student's structured low-level output and content-dependent reconstruction error. More history or a universal waveform shift is not supported as the main remedy by these results. No parameters, optimizer state, training exposure or historical quality scores were changed.
