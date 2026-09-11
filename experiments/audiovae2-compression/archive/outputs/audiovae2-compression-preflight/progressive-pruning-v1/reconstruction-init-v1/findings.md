# Reconstruction-aware pruning: measured results

The additional startup-pattern comparison and two initialization experiments are complete. Reconstructing the retained weights substantially improves the fresh pruned student. It does not yet solve startup silence or outperform the recovered 5,000-step checkpoint.

## What was actually removed

This cut narrows stage2 from 512 to 384 channels. It does not delete stage2. Stage3 remains 256 channels, and all nine residual units across stages2–4 remain. Four mixing operations lose inputs: three stage2 residual projections and the stage3 upsampler.

The previous selector ranked channel features and then sliced weights. It did not reconstruct the retained weights to compensate for removed inputs. The new fits address that missing initialization step without adding model operations or changing the causal geometry.

## Aggregate startup evidence

Across ten exactly zero-input starts, more than 99.999999% of the residual energy is captured by a shared mean startup pattern. The 5,000-step model's across-source disagreement is approximately 1.53e-9 RMS, versus 2.26e-5 pooled residual RMS. The residual peak occurs at 0.8125ms in each of these ten cases. This supports a repeatable network startup response, not a language-specific fluctuation.

In the prior fresh-step0 causal intervention, restoring the omitted teacher contribution only at the stage3 upsampler reduced startup residual RMS from 48.385 to 5.151 microFS and made all 13 startup checks pass. Restoring all four omitted contributions recovered near-numerical parity. These were diagnostic interventions using unavailable teacher features, not deployable fixes. They cannot be inserted unchanged into the coadapted 5,000-step model.

## Native weight reconstruction experiment

Both independent candidates start from the same original fresh slice. A refits only the stage3 upsampler. B sequentially refits the three stage2 residual projections, recomputing current student inputs each time, then refits the upsampler. Each uses all valid rows from the same 72 calibration sources. The original 96-source development panel and thresholds remain unchanged.

Corrections are fitted in FP64 and installed in the existing FP32 weights and biases. Residual targets account for the actual current skip path; the upsampler retains its five phases, current/previous context and one shared bias.

| Metric | Fresh sliced initialization | A: upsampler refit | B: four refits | Preserved 5,000-step model |
|---|---:|---:|---:|---:|
| Active waveform cosine | 0.743507 | 0.953530 | 0.965930 | 0.995233 |
| Waveform MAE | 0.021871 | 0.007477 | 0.006623 | 0.001964 |
| Mel error | 1.029526 | 0.268096 | 0.222572 | 0.126230 |
| Stage2–4 output MSE | 0.046927 | 0.003265 | 0.002496 | 0.001463 |
| Quiet residual RMS, microFS | 1414.630 | 166.773 | 136.315 | 70.263 |
| Quiet checks passing / 2544 | 0 | 502 | 751 | 1511 |
| Near-silence startup passing / 13 | 0 | 0 | 0 | 0 |
| Startup residual RMS, microFS | 48.385 | 12.272 | 27.966 | 22.004 |
| Other near-silence passing / 171 | 0 | 166 | 171 | 171 |
| Maximum output peak | 0.080676 | 0.980647 | 0.988390 | 0.992325 |
| Gradient-based updates | 0 | 0 | 0 | 5000 |

microFS means one millionth of full scale. Correlation is teacher-active waveform cosine, not perceptual accuracy. The quiet set includes low-level nonzero audio, not only physical silence. Startup and other near-silence groups are disjoint; another 2360 quiet windows have teacher RMS above1e-5. These development recordings are disjoint from fitting but have been used for earlier experiments; they are not an untouched final test set.

B reduces initial waveform MAE by69.72%, group MSE by94.68% and quiet RMS by90.36%. A and B both improve waveform MAE, mel and group MSE on all96 recordings versus the fresh slice; cosine improves on all94 recordings for which it is defined. B beats A on waveform MAE for84/96 and mel for96/96. Both remain worse than the preserved5,000-step model on waveform MAE and mel for all96 recordings.

Thus B is a promising initialization, not a promoted model or proof of faster convergence. A is stronger on the specific startup metric. B still passes all171 other near-silence windows; its13 startup failures do not mean silence fails everywhere.

## What remains unresolved

All13 startup windows still fail for both refits. A cuts their residual RMS by74.64%; B cuts it by42.20%. Better pooled feature reconstruction therefore does not guarantee a better startup waveform.

Calibration contains6 near-silent startup windows, including5 with exactly zero source reference. Their5760 samples constitute0.0697% of8266341 valid calibration samples. This is consistent with rare startup trajectories receiving little influence in pooled fitting, but sample share is not a measurement of feature-error or gradient share and does not prove the cause.

Native writeback, solver residual and baseline checks passed. The upsampler fit is full rank with regularized condition around1e6 and normal-equation residual below3.3e-15. This is not proof of statistical generalization or that a linear correction can express every missing startup contribution. The held-out waveform results, not fit error, determine usefulness.

The earlier identical-teacher control passed60000 crops and more than1.15million quiet windows. The current failures are therefore not explained by the wrapper universally inventing relative silence error. This also does not claim the original teacher emits physical zero for every zero-input interval.

## Recommendation

Keep the5,000-step checkpoint and both fitted candidates. Do not make another width cut yet.

The next bounded test should be one startup-constrained version of A. Fit its existing upsampler to ordinary calibration targets while constraining the observed calibration startup interface to preserve the teacher response. Use only calibration data, the native shared bias and causal phases. Explicitly report rank, compatibility and FP32 constraint residual; if exact constraints are infeasible, report that rather than silently loosening them or choosing a large quiet weight.

Evaluate the same96 full waveforms, startup, other near-silence, quiet nonzero audio, mel and active error. Hidden-interface equality is a sufficient diagnostic target, not a necessary condition for perceptual quality. Failure of this particular constrained fit would not prove the student width is insufficient.

After a suitable initialization is established, test bounded joint recovery of stages2–4 against the fixed original teacher, measuring pure-student outputs. This is the appropriate point to compare convergence with the old sliced start. A slower schedule alone remains unproven.

For later cuts, use a structured sequence: choose channels based on downstream reconstruction, refit the affected native operators, recover jointly, and review startup and ordinary audio separately before the next cut. A gradual training-only transition remains a fallback if abrupt cuts still produce a recovery gap; its teacher assistance must reach zero, with pure-student evaluation throughout.

## Literature connection

[StreamCodec2](https://arxiv.org/html/2509.13670v1) supports intermediate teacher guidance. It is not evidence that slow distillation guarantees teacher parity.

The closer precedents are [Channel Pruning](https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html), which reconstructs layer outputs after selection, and [asymmetric reconstruction](https://arxiv.org/pdf/1505.06798), which uses actual compressed inputs to predict original outputs. The current audio adaptation produced the measured improvements above. Their image-model results do not establish audio quality or CPU RTF.

[Asymptotic Soft Filter Pruning](https://arxiv.org/abs/1808.07471) provides a gradual-removal precedent. That is different from merely reducing learning rate. It has not yet been tested on this cut.

No new inference layers or tensor dimensions were introduced. Actual CPU RTF was not benchmarked in this diagnostic, so no new speedup is claimed. No further gradient training, promotion, commits or push occurred.

