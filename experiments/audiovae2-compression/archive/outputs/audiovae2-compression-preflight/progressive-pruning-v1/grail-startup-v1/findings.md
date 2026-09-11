# Fresh G plus startup initializer: qualified

11 September 2026. Fresh original teacher factory, original sealed G native fits, then actual-G-feature reconstruction and startup equalities in the existing stage-3 upsampler. No trained checkpoint and no neural updates. Three G residual mixers and frozen outer stages are unchanged; no inference operation was added.

| Metric | Untrained G | Fresh G plus startup |
|---|---:|---:|
| Calibration startup passing | 0/6 | 6/6 |
| Development startup passing | 0/13 | 13/13 |
| Waveform MAE | 0.00802214 | 0.00700703 |
| Active waveform cosine | 0.955348 | 0.961949 |
| Mel error | 0.249693 | 0.237512 |
| Quiet windows passing | 567/2544 | 597/2544 |
| Startup residual RMS, millionths full scale | 9.280 | 2.035 |
| 20–40ms transient passing | 0/10 | 0/10 |
| Full-scale overshoot samples | 0 | 0 |

Waveform MAE improves12.65% versus G initialization, with89/96 recordings improved. The fit also improves ordinary hidden-output reconstruction, so this is not a startup-only ablation. The upsampler now leaves G's shared zero-intercept map family; label it G plus a constrained native upsampler. No near-perfect fidelity claim follows from these initial results.

Original G's complete96-source baseline reproduced; all336 main teacher-cache comparisons and six separate anchor preparations were exact. The fixed native solver was feasible and all six FP32 hidden equalities passed after writeback. All file/state/RNG guards passed. Original72 calibration and96 development sources remained separate. Development observations did not influence the fit.

Initializer computation elapsed32.23s, excluding initial loading/authentication. Two earlier import-only attempts stopped before computation; their logs remain retained. The corrected dependency launch succeeded.

Qualifies the matched fresh64 ordinary/retention pilots. The2,000-update recovery is approved after that pair shows useful learning; it has not started. [Aggregate evidence](completed.json).
