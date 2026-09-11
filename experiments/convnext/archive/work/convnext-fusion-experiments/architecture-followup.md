# Why the architecture changes did not remove the quiet floor

Read-only mathematical review of saved weights, results and source, 2026-09-09. No model inference, training, remote changes or new benchmark was performed. Filter responses below are calculations from the seven already saved trained coefficients.

## What is diagnosed

1. The terminal tanh did eliminate above-full-scale samples in the observed panel, but it did not solve quiet reconstruction. At amplitude 0.0002, `tanh(x)/x = 0.9999999867` and `d tanh(x)/dx = 0.99999996`. There is essentially no small-signal attenuation. Near-full-scale compression is a different effect, which explains why recovery of loud transients still requires adaptation.

2. The trained filter did not learn a noise-removing response. Its taps, oldest to current sample, are:

   `[0.00090633356, 0.00081026595, 0.00073069142, 0.00061103463, 0.00102364784, 0.00171782612, 0.99737650156]`.

   Their sum is **1.00317630108**, so the filter slightly amplifies DC. At 48 kHz, `H(f) = sum(k=0..6, w[k] exp(-j 2 pi f (6-k) / 48000))` gives:

   | Frequency | Magnitude | Gain, dB |
   | --- | ---: | ---: |
   | DC | 1.00317630 | +0.027545 |
   | 100 Hz | 1.00316983 | +0.027489 |
   | 200 Hz | 1.00315044 | +0.027321 |
   | 1 kHz | 1.00255188 | +0.022137 |
   | 8 kHz | 0.99805922 | -0.016874 |
   | 16 kHz | 0.99675280 | -0.028251 |
   | 24 kHz | 0.99689805 | -0.026985 |

   This is effectively a near-identity response across the checked frequencies. Its seven samples cover 0.14583 ms; the oldest-to-current delay span is 0.125 ms. Even a uniform seven-tap moving average would have magnitude 0.999657 at 100 Hz. A short filter can be designed to reject DC or a frequency, so the claim is not that seven taps make rejection mathematically impossible. The actual trained coefficients simply do not perform that rejection.

   The 0.32% DC amplitude gain cannot alone explain the observed roughly 10.5% natural quiet-RMS regression. The whole decoder was fine-tuned, so its changed synthesis trajectory also contributes. The saved results do not isolate the filter's direct effect on the original unchanged decoder.

3. Startup padding is not the cause of mature quiet residual in these checks. The earlier zero-padding ablation produced exactly unchanged mature output once complete history had passed. On the six-second encoded-zero fixture, the final two-to-six-second region still has substantial residual. All six trained decoders also passed sample accounting and batch/streaming checks. Their encoded-zero batch/stream errors are around 1.7e-8, whereas steady teacher-relative residual is around 2e-4. That is not consistent with these chunk-state discrepancies explaining this floor.

4. The quiet error is not only an unfavorable waveform correlation statistic. The saved control's steady teacher RMS is **9.62952e-6**, while student RMS is **2.10760e-4**, a **21.89x amplitude ratio**, or about **26.8 dB higher**. These correspond to approximately -100.33 and -73.52 dBFS. This is measurable excess energy. It is not a calibrated audibility assessment.

## What the periodic metric does and does not mean

The metric folds quiet audio into 480-sample pieces and averages each phase. At 48 kHz, a repeated 480-sample template has a 10 ms period, fundamental spacing 100 Hz, and can contain DC plus harmonics at 100, 200, 300 Hz and higher. Template RMS does not say where that energy is located. No FFT or spectral-energy decomposition of the residual template was saved in this task, so identifying a 100 Hz tone from this metric would be an overclaim.

The model's four-phase adapter also distinguishes four internal frames per raw latent. A steady four-phase internal cycle could produce a 1,920-sample/40 ms pattern, with 25 Hz harmonic spacing. Averaging only 480-sample pieces can hide differences between those four phases. The existing metric is a diagnostic projection of the error, not a complete description of its temporal structure.

## Why shorter mel windows did not directly address this mechanism

The new FFT sizes 256 and 512 cover 5.33 and 10.67 ms, with FFT-bin spacing 187.5 and 93.75 Hz. They improve temporal localization relative to the previous longer windows, but are not precise low-frequency or DC-separation tools. Existing 1024, 2048 and 4096 windows already have 46.875, 23.4375 and 11.71875 Hz bin spacing.

All five mel terms compare spectral magnitudes, and mel pooling discards additional spectral detail. They do not impose phase equality or an explicit small-signal output bound. The raw waveform loss still supplies signed sample-aligned supervision, so there is not a complete absence of phase gradients. The logarithmic term has zero derivative where its own magnitude clamp is active, but the linear mel and waveform branches remain present; this does not establish globally zero quiet gradients.

The candidate also redistributed the fixed mel branch budget: the two new scales received 2/5 of the equally averaged spectral objective. It was a change in temporal resolutions and relative weighting, not an isolated test of whether any possible short-time supervision can help. The screen gives no evidence that this particular redistribution targets the mature quiet-response defect.

## What the head permits

After the causal body and PReLU head, each internal frame has a feature vector `a_t` with 2,048 components. The final bias-free projection produces `p_t = W a_t`, with 480 output positions. Those output positions share features and losses but have different weight rows. There is no architectural requirement that a low-energy input regime produces the teacher's low-energy waveform.

For a roughly steady quiet regime with four internal phase features `a_0 ... a_3`, the relevant target condition is approximately `W a_phase = teacher_quiet_phase`. It is not `W 0 = 0`. The encoder output for real zero audio is nonzero: an earlier saved six-second teacher diagnostic had latent RMS about 0.653. The teacher output is also not mathematically zero. Removing a final bias cannot solve this; that projection already has `bias=False`, and earlier affine terms and nonlinear features remain.

The final projection could in principle learn small outputs in these feature directions without adding inference operations. We have not inspected the actual trained quiet head features or measured their conditioning, so neither a head-capacity defect nor a particular affine/bias parameter is established as the cause. Centering latents at an estimated silence vector is a reparameterization or a behavior change, not an automatic proof of a correct null response.

## A more targeted, still architecture-compatible next direction

The next candidate should explicitly teach the desired **teacher-relative quiet response**, rather than add another generic layer. Keep the encoder, 64-channel contract, current body and sample accounting.

- Use independently prepared training-only quiet/near-silent examples with actual frozen-encoder latents and frozen-decoder targets. Include varied preceding histories and soft speech/whisper transitions. Preserve the existing held-out synthetic fixtures; do not train on the exact evaluation fixture.
- Retain sample-aligned waveform reconstruction. A bounded quiet-region contribution or a teacher-relative local energy ceiling would target the observed excess energy directly. The ceiling should follow the teacher's actual local RMS plus a declared absolute tolerance, not force every quiet output to zero or depend on tiny inverse-RMS divisors. Teacher-relative residual remains necessary because matching RMS alone can still produce the wrong waveform.
- A diagnostic head-only update, with the body fixed, could distinguish whether the existing output projection can satisfy quiet anchors while retaining representative speech outputs. That would test representational compatibility before committing to another full-network recipe change. No such update or feature extraction was performed here, and it is not a demonstrated solution.

These are training-only constraints or restricted fitting procedures. They add no decoder inference work if adopted as losses or learned weights. A hard inference gate, unconditional silence-template subtraction or an arbitrary DC blocker would modify breaths and legitimate low-frequency speech too, and is not justified by these results.

The concrete missed point is that the tested changes did not directly enforce the existing decoder's response to the teacher's quiet latent region. Tanh controls the top of the amplitude range; the learned filter barely changes any frequency; startup changes only boundaries; shorter magnitude windows refine another view of the same reconstruction task. Which optimization or feature-direction mechanism prevents quiet waveform supervision from succeeding remains unmeasured. That distinction should guide the next diagnostic instead of claiming that a complete new architecture or more steps alone is the answer.

## Saved evidence

- `outputs/convnext-fusion-experiments/streaming-validation.json`: trained filter taps, steady teacher/student RMS, all 36 streaming comparisons.
- `outputs/convnext-fusion-experiments/report.md` and `results.json`: matched candidate quality measurements and their scope.
- `outputs/convnext-remediation/encoded-silence-audit.json`: actual encoded-zero latent magnitudes and the teacher's nonzero output behavior.
- `audiovae_student/model.py`, `fusion_evaluation.py`, `reconstruction_v2.py`: head structure, phase-template definition and exact reconstruction reductions.
