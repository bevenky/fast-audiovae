# Decoder refinement results

The original run remains paused at step 8,090. Six independent branches each completed 200 generator updates from the same preserved checkpoint. The two fresh-discriminator branches also completed 20 discriminator-only updates each. All six final checkpoints and their optimizer states are retained on Runpod. No production model or Git commit was changed.

A follow-up audit found two important limitations: the newly changed losses inherited stale gradient-normalization statistics, and named edge cases were scarce in the selected training subset. The complex discriminator therefore received far less perceptual supervision than intended. Its architecture remains inconclusive. Keep the trained base and fix those experiment controls before choosing a combined recipe. Tanh is still a useful bounding candidate, but this screen does not establish quality-neutral adoption.

## Matched quality comparison

These are teacher-reconstruction measurements, not perceptual MOS scores. The common before/after quality panel ran on the H100 training device. CPU startup and streaming checks were separate; no GPU timing is presented as CPU RTF. Natural-audio results use 281 held-out crops from 143 sources. Three separately reported synthetic fixtures bring the complete panel to 284 crops. Speech and expressive correlation are means over active crops; the expressive category is limited to explicitly classified cases. Raw error is weighted by valid samples. Mel is the unchanged common diagnostic averaged over crops. Quiet residual is pooled over natural quiet windows. Lower errors are better; higher correlation is better.

| Branch | Speech correlation | Expressive correlation | Raw error | Fixed mel error | Natural quiet RMS | Maximum peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged control | 0.95548 | 0.74674 | 0.012199 | 1.1878 | 0.000347 | 1.299 |
| Terminal tanh | 0.95502 | 0.74434 | 0.012259 | 1.1798 | 0.000352 | 0.922 |
| Short-window mel | 0.95536 | 0.74627 | 0.012228 | 1.1809 | 0.000362 | 1.297 |
| Seven-tap output filter | 0.95499 | 0.74637 | 0.012215 | 1.1862 | 0.000383 | 1.304 |
| Fresh magnitude discriminator control | 0.95524 | 0.74605 | 0.012212 | 1.1870 | 0.000360 | 1.306 |
| Complex multiband discriminator | 0.95526 | 0.74670 | 0.012201 | 1.1839 | 0.000367 | 1.295 |

The complex candidate must be compared with the fresh magnitude control. Both retain the previously trained period discriminators and receive the same additional discriminator-only exposure. Comparing a newly initialized spectral discriminator only against the fully trained original would confound architecture with training age.

## Decisions

| Change | Evidence versus its matched control | Decision |
| --- | --- | --- |
| Terminal tanh | Zero overshoot across the panel. Natural waveform error increased 0.50%, quiet residual increased 1.68%, and fixed mel error decreased 0.68%. | Retain as the next adaptation candidate. Do not claim quality-neutral adoption yet. |
| Short-window mel | Quiet residual increased 4.32%; waveform error increased 0.24%. Fixed mel error improved 0.58%, without a clear transient-quality win. | Recalibrate the changed loss scale and verify targeted data before judging this weighting. |
| Seven-tap causal output filter | Quiet residual increased 10.52%; waveform error increased 0.14%. It did not remove overshoot. | Do not include this candidate. |
| Complex multiband discriminator | Versus the fresh magnitude control, waveform error improved 0.09% and mel improved 0.26%, while quiet residual increased 1.86%. | Inconclusive: stale normalization strongly suppressed its new perceptual gradients. Correct that before retesting. |
| Zero startup padding | Valid-prefix error improved 2.68%, but mature output was exactly unchanged and pooled quiet error in the startup-crop panel increased 23.67%. | Keep the current padding. |

Tanh reduced the maximum student peak from 1.299 to 0.922; the teacher maximum on this panel is about 0.995. A lower bounded peak is not by itself evidence of faithful transient reconstruction. Speech correlation changed from 0.95548 to 0.95502, and expressive correlation from 0.74674 to 0.74434. Natural waveform error improved on 100 crops and regressed on 181. Generic emotional/nonverbal material had a 2.25% waveform-error increase; laughter had a 0.88% increase. These are small but real measured tradeoffs, so the zero-loss requirement has not been demonstrated.

None of the candidates passed the existing quiet-audio checks: all 3,336 natural quiet windows still failed. Encoded digital silence also remains nonzero. These are the existing engineering thresholds, not a measured audibility verdict.

## Implementation and controls

- Kept the frozen AudioVAE2 encoder, raw 64-channel latents and post-tanh teacher waveform targets unchanged. Teacher targets were prepared once and reused across all branches.
- Preserved all original Muon and AdamW moments, calibrated normalization, global step and crop RNG. Only the filter gained a new seven-parameter AdamW group. Fresh spectral heads received fresh optimizer state, while learned period-discriminator weights and moments were preserved at the fork.
- Used the same 6,400 nonoverlapping scored windows once per generator branch. The discriminator pair used a separate 640-window warmup pool. Reuse across independent debugging branches was deliberate; there was no repeated scored window within an arm.
- Supplied 30 actual preceding latent frames for training, sufficient for the extra six waveform-history samples in the filter. On archived evaluation crops with shorter available context, the first six scored samples were excluded uniformly across all branches. This removed 618 evaluation samples in total and did not change streaming output or sample counts.
- Short-window mel added FFT 256 and 512 to the existing 1024, 2048 and 4096 scales. Each of the five scales had equal weight within the unchanged total mel branch budget. This tests both shorter analysis and redistribution of scale weighting.
- Complex discrimination kept raw amplitude, real and imaginary STFT components, and five frequency bands. No per-clip peak normalization or DC removal was introduced.
- No combined architecture branch, new silence penalty, learning-rate change, automatic normalization recalibration or extra long training run was started.

All 78 focused candidate, migration and evaluation tests passed on Runpod, including native Muon. Independent report checks verified the exact parent, final model tensors, architecture identities, teacher/crop identities, valid sample masks and checkpoint hashes for all six branches.

## CPU and streaming limits

The initial one-thread CPU screen used the unfused PyTorch student on the Runpod AMD CPU, 80 ms chunks, one two-second latent sequence, and seven randomized-order repetitions. It is an output-layer overhead check, not the optimized ONNX/native deployment RTF or an Intel/Apple benchmark.

| Initial architecture | Unfused streaming RTF | Batch/stream maximum difference | Samples returned |
| --- | ---: | ---: | --- |
| control | 0.2137 | 8.94e-07 | 96,000 / 96,000 |
| tanh | 0.2111 | 8.05e-07 | 96,000 / 96,000 |
| filter | 0.2119 | 8.94e-07 | 96,000 / 96,000 |
| zero_padding | 0.2116 | 8.34e-07 | 96,000 / 96,000 |

The roughly 1% variation between these measurements does not establish a speedup. No meaningful output-layer slowdown appeared in this limited screen. Training-only reconstruction losses and discriminators add no decoder inference work. Folded/exported variants remain unqualified, particularly for zero-padding boundary semantics.

## Trained streaming verification

All 36 checks on the actual trained checkpoints passed using one CPU thread: six branches, encoded silence plus held-out speech, and 80 ms, 160 ms and irregular chunks including empty calls. Every chunk emitted the expected samples; empty calls preserved state. The largest batch/stream difference was 9.24e-7, below the fixed 2e-6 tolerance. No waveform was trimmed to obtain this agreement. [Detailed streaming evidence](streaming-validation.json).

## Next action

Keep the step-8,090 original and all experimental checkpoints. Start the corrected comparison from this trained base, preserving its model and optimizer state. Recalibrate only the gradient-normalization statistics whose loss definitions or discriminators changed, and verify achieved contributions before judging quality. Build a training subset with explicit quotas for the failing sound categories. Benchmark teacher-relative quiet behavior and phase structure before proposing another layer. The original run remains paused; a corrected pilot has not started. See [follow-up audit](follow-up-audit.md) for the verified gaps and next plan.

The detailed paired measurements, including synthetic fixtures and worst language/event groups, are in [results.json](results.json). This was a single-parent, single-seed, 200-update screen. It establishes the observed short-run tradeoffs; it does not establish statistical significance, final convergence, listening quality or a production RTF improvement.
