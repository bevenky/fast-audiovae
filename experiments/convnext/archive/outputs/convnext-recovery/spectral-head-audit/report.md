# Remaining reconstruction gaps and AudioVAE2 transfers

The current joint-head spectral candidate is retained unchanged. The strongest confirmed limitation of the last repair experiment is its conflicting targets and restrictive selection rule. Those controls were appropriate for selective silence repair, but cannot be carried unchanged into a final teacher-reconstruction recipe. The results do not demonstrate that the inexpensive decoder architecture has reached its limit.

This audit inspected saved measurements, training and selection code, earlier diagnostic evidence, and official upstream sources. It executed no model forward pass, optimizer update or performance benchmark. The only remote changes were verified archival copies of the candidate, its parent checkpoint and results into `/workspace`.

## What is actually preventing further improvement?

### The repair objective protects errors we eventually need to correct

The latest objective matches AudioVAE2 waveforms inside quiet regions, but outside them it penalizes changes from the original student, with a preservation coefficient of 100. Teacher mel supervision applies everywhere. Consequently, the waveform and spectral terms can request different outputs outside silence. A teacher-perfect waveform has zero teacher reconstruction error but nonzero old-student preservation error. This is a confirmed target mismatch; it does not establish which branch dominated each individual failed recording.

The selection rule makes the distinction even clearer. It requires nonquiet displacement MSE from the old student to remain below 1% of that student's original teacher-error MSE. Exact teacher matching would have a ratio of 1 and fail a limit of 0.01. Along a straight interpolation toward the teacher, this rule permits only 10% of the full displacement. A final continuation should protect teacher-relative quality rather than require the waveform to remain nearly identical to the old student.

This limitation concerns the latest restricted head experiment. It is not a claim that the earlier full-model run lacked teacher waveform supervision. It also does not invalidate the matched result showing a benefit from adding spectral supervision.

Evidence: [objective selection](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-recovery/architecture-experiments/run_spectral_heads.py:243), [waveform targets](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-recovery/architecture-experiments/run_joint_heads.py:88), [selection rule](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-recovery/architecture-experiments/run_joint_heads.py:188).

### Stationary silence and natural quiet require different reconstruction

The candidate reduces stationary residual RMS by 49.2%, but natural quiet RMS by 2.0%. The earlier decomposition of its parent checkpoint found that 81.9% of natural quiet residual power varied with the input. That number was not remeasured on this candidate. It explains why canceling a constant repeating pattern cannot be assumed to solve breathing, low-volume speech or changing background sounds.

All 87 natural sources with waveform-scored quiet samples improve in quiet waveform error. However, intact quiet spectral windows exist for only 79 sources, and five have more than 1% local mel regression. These are different measurements and different temporal support.

| Local quiet region | Linear mel-error change | Log mel-error change | Available frames at FFT 1024 / 2048 / 4096 |
| --- | ---: | ---: | --- |
| Screaming source `freesound:220655` | +1.28% | +11.96% | 8 / 2 / 0 |
| `freesound:220663` | −6.36% | +4.32% | 12 / 4 / 0 |
| Manipuri `6473924464383473_chunk_1.flac` | +2.07% | +4.87% | 4 / 0 / 0 |
| `emogator:000120-12-1.mp3` | approximately unchanged | +1.75% | 79 / 34 / 12 |
| `emogator:000282-30-3.mp3` | −6.28% | +1.17% | 91 / 42 / 17 |

The strongest local regressions have sparse support, sometimes across overlapping crops. They are not independent events or proof of equivalent audible deterioration. Nevertheless, the log term exposes worse relative spectral detail even where absolute magnitude error improves. Raising the log floor merely to hide these differences would not be a reconstruction fix. Saved aggregates cannot establish whether particular bins crossed the floor.

The current mel loss pools all valid time-frequency elements; quiet, active and transition partitions are evaluation-only. Thus improvement over a long active region can conceal deterioration in a brief quiet one. Short quiet intervals are not discarded from the full-waveform STFT: they can occur inside mixed windows. All selected training and validation scored spans accommodated the largest FFT, so the short-span omission branch did not remove training examples in this run.

Evidence: [pooling](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-recovery/architecture-experiments/run_spectral_heads.py:63), [region measurements](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-recovery/architecture-experiments/run_spectral_heads.py:119), [parent decomposition](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/peak-silence-diagnosis/report.md).

### The two active-speech failures are real tradeoffs, not absent languages

Using the pooled spectral metric, Bodo and Manipuri worsen by 1.56% and 1.58%; the original equal-crop source metric gives 1.24% and 1.25%. Both their linear and log components worsen. Bodo's evaluated spectral windows are entirely active; the Manipuri failure is predominantly active. Both sources also fail under projection-only spectral adaptation, so the nonlinear head is not uniquely implicated.

The matched fit includes 26 Bodo and 17 Manipuri recordings. Their absence from training is therefore ruled out. Shared weights and the competing targets are plausible contributors, but identifying the contribution on each source requires an actual gradient/update decomposition. It would be incorrect to name a defective layer from these metrics alone.

### The latest experiment was not a peak-reconstruction treatment

Its nonquiet waveform reference is the old student, which already overshoots. Magnitude mel supervision does not uniquely specify sample phase or peaks. The candidate has no output bound or explicit peak term. It is therefore unsurprising that maximum amplitude falls from 1.296632 to 1.292609 while scored overshoot observations rise from 588 to 589.

Earlier diagnostics located interior, short amplitude errors rather than a consistent block-seam problem. They also demonstrated optimizer-induced adverse directions on one full-model update. That older attribution cannot be assigned to this fresh-AdamW, head-only experiment without measuring it here.

Restoring teacher waveform targets outside silence supplies a missing direct correction in this experiment. If rare peaks still trade against ordinary samples, the next training-only option is event-focused teacher reconstruction with intact context and controlled sampling, followed only if needed by a range penalty. Such a penalty is our adaptation of the bounded teacher target, not a verified AudioVAE2 training method. It cannot guarantee bounded output on unseen inputs.

### Calibration and data coverage still need explicit controls

The spectral coefficient 8.0579 was calibrated on four initial batches. At that point, predictions equal the old student and preservation gradients are zero. Per-batch equalizing coefficients ranged from 3.99 to 18.35. Initial waveform/spectral gradient cosines were small positive values, 0.022–0.074. These facts do not demonstrate gradient conflict or dominance at initialization.

Once the head moves, preservation gradients become active. The saved logs contain component losses and a combined gradient norm, not later component directions. A next bounded comparison needs a small fixed training-only panel to inspect quiet, active and spectral gradients plus the actual optimizer displacement before and after fitting. Do not infer gradient balance from numerical loss values or blindly reuse the preservation coefficient as a teacher-loss weight.

The current fit has 2,048 distinct sources and 83.6 scored minutes, including 667.54 seconds of teacher-defined quiet audio, or 13.31% of samples. The saved selection identity matches the latest run. There are multilingual and expressive-source datasets, including three dedicated human-whistle sources. However, every selected row's `condition` field is null. Dataset membership alone cannot prove that a particular scored crop contains laughter, crying or another requested event. Restore source-label joins and verify event intervals before claiming coverage; keep held-out failures out of fitting. The existing panel remains development data, not a fresh final test.

## What we can take from AudioVAE2 without adding deployed work

The verified decoder implementation uses weight-normalized causal convolutions and a terminal Snake, kernel-seven convolution and tanh. The current student's ordinary head convolutions lack weight normalization. The public V2 report does not establish a complete checkpoint-specific training recipe, so DAC settings below are explicitly lineage references rather than claimed V2 settings. [AudioVAE2 implementation](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/modules/audiovae/audio_vae_v2.py), [VoxCPM2 report](https://arxiv.org/html/2606.06928v1#S3.SS2).

| Technique | What it could address | Deployed CPU effect | Decision |
| --- | --- | --- | --- |
| Train the existing head with weight normalization | Separates each weight row's magnitude from direction and changes optimization geometry | Materialize the effective weights and remove parametrization before export; same convolutions and weight shapes | Best additional technique directly verified in AudioVAE2; test separately after objective correction |
| Shorter multiscale spectral supervision from DAC lineage | Gives brief quiet transitions and transients more localized supervision | Training only | Next loss candidate; preserve current long-scale supervision and score unchanged metrics |
| Complex multiband discriminator and feature matching from DAC lineage | Additional phase-sensitive and frequency-local perceptual feedback | Training only | Already implemented experimentally; prior spectral gains came with quiet/peak tradeoffs, so not a missing universal fix |
| Jointly adapt the last existing block with the head | Allows upstream features to separate quiet and active trajectories | Same deployed graph | Our lower-priority capacity/optimization test if corrected head fitting stalls |
| Selected intermediate teacher-feature targets | Can guide the existing student feature path with auxiliary training projections | Remove auxiliary projections at export | Distillation proposal, not verified as V2's recipe; first justify temporal and channel alignment |

For weight normalization, initialize the training representation from the saved weights rather than reinitialize the head: use each existing row as its direction and its norm as its gain. Materialize trained weights for deployment. This is mathematically the same class of convolution, not added inference capacity or a promised quality gain. Zero-norm rows and finite-precision initialization/folding need explicit parity checks, and changed optimizer parameterization needs a controlled comparison. [Local teacher wrappers](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/audio_vae_v2.py:41), [current student head](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/fast-audiovae/experiments/convnext/audiovae_student/model.py:231), [weight materialization implementation](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/work/convnext-natural-context-repair/upstream-source-audit/weight_norm.py:70).

Our shortest mel window is 1,024 samples, or 21.3 ms at 48 kHz. DAC's released configuration uses windows from 32 to 2,048 samples at 44.1 kHz. Adding shorter analysis is a training-cost change, not decoder computation. Earlier short-window trials redistributed the existing scale budget and had other experimental limitations, so they do not prove a clean isolated short-scale addition is ineffective. Do not transplant every scale, weighting or waveform normalization blindly. [DAC configuration](https://github.com/descriptinc/descript-audio-codec/blob/main/conf/base.yml), [earlier experiment limitations](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-fusion-experiments/report.md).

DAC's released discriminator includes complex spectral inputs and frequency bands. Our corrected experiment already tested that family and found a spectral benefit with unresolved quiet and peak tradeoffs. It should remain available without adding a fresh discriminator and changing the head objective simultaneously. [DAC discriminator](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/discriminator.py), [corrected comparison](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-corrected-experiments/report.md).

AudioVAE2's tanh provides a hard range bound, but adding tanh is not literally free CPU work. A training-only range objective can encourage bounded outputs without that operation, but cannot provide the same guarantee. The teacher's high-rate Snake/residual stack and final sample-rate convolution also do not satisfy the unchanged-operations requirement. They are not recommended for this next step. KL regularization, new latent jitter and re-encoding losses are unnecessary for correcting an already shared frozen 64-channel latent interface.

## Recommended next sequence

1. Preserve the current candidate and use it as the common starting point for separate follow-up copies. Keep the original step 8,890 control too.
2. Correct the nonquiet training target and teacher-relative selection contract first. Keep the head topology, body, normalization, latent tensors and existing spectral windows unchanged. Calibrate teacher-directed terms on training data rather than carrying over the old 100× preservation weight. Track local quiet/transition spectra and per-source regressions alongside whole-recording averages.
3. Under the corrected objective, compare ordinary head fitting with weight-normalized head fitting from the same saved weights and data. This is the cleanest additional AudioVAE2 decoder technique with no deployed operation increase.
4. If short quiet/transient failures remain, isolate shorter spectral supervision. Preserve actual waveform targets, contiguous context and the original evaluation definition. Restore event metadata and verify event coverage before fitting.
5. Only if those results establish a limitation should existing late-block adaptation or auxiliary teacher features enter another comparison. A larger decoder or additional nonlinear layer is not supported by current evidence.

These are proposed tests, not started work or guaranteed fixes. The current candidate remains unpromoted because it still fails quality checks. No CPU timing claim follows from this audit; unchanged inference operations are a design constraint to preserve, with numerical and streaming parity to verify after any eventual training/export change.

## Retained artifacts

The candidate, full parent checkpoint and experiment reports were copied with SHA-256 verification to `/workspace/fast-audiovae-convnext-20260909-r9/retained-candidates/joint-spectral-step8890-256-20260910` on Runpod. Original files were unchanged. The candidate head hash is `b612baf57c5a3ae113ed67ea1a48f6f9cb2d740001aa5b2b5dc3a7a5ba9097e2`; its parent hash is `f7a91a2f3bef6fdec2a37643dc661a97b18a24b9e25b822df49c1c4017ddc948`.

[Preservation receipt](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/spectral-head-audit/preservation-receipt.json) and [recomputed audit measurements](/Users/venky/Documents/Codex/2026-09-07/https-arxiv-org-html-2402-10533v2/outputs/convnext-recovery/spectral-head-audit/measurements.json) accompany this report. No model source, weights, loss definitions, checkpoint selection thresholds or production runtime were modified.
