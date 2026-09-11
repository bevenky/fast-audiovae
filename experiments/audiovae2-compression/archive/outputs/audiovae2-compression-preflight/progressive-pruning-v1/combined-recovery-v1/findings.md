# Combined B plus startup initialization: completed recovery

The combined candidate C completed 2,500 updates, but the result is mixed. Its startup correction was lost during unrestricted joint recovery. Against B at the same 2,000 updates, C improves waveform error and correlation while worsening mel, group-feature and quiet metrics. Against the original initialization at 2,500 updates, C improves mel and group-feature error but worsens waveform error and quiet performance. These results do not establish an overall replacement or a durable startup fix.

This is an independent arithmetic and consistency review of the local aggregate view. The remote CPU audit checked checkpoints, source ledgers and captured windows; no new inference or training was performed for this review. Source IDs, tensors and audio remain remote.

**Matched C2000 versus B2000.** Both executed the same ordered first 24,000 distinct crops, with the same teacher, architecture, losses, learning rate and accumulation. Percentage changes below are C relative to the reference; correlation changes are percentage points. Lower error is better.

| Metric | B2000 | C2000 | Change |
|---|---:|---:|---:|
| Waveform MAE | 0.00261465 | 0.00249149 | −4.710% |
| Waveform MSE | 0.0000769228 | 0.0000718074 | −6.650% |
| Waveform NRMSE | 0.147290 | 0.142309 | −3.382% |
| Mel error | 0.141059 | 0.142995 | +1.373% |
| Group-output MSE | 0.00148276 | 0.00151621 | +2.256% |
| Group-output NRMSE | 0.109615 | 0.110845 | +1.122% |
| Active waveform correlation | 99.237506% | 99.294686% | +0.057180 pp |
| Quiet residual RMS, micro full scale | 78.7338 | 83.0031 | +5.422% |
| Quiet windows passing, of 2,544 | 1,242 | 643 | −599 |

At C2000, paired MAE improves on 75 sources and worsens on 21; correlation improves on 74 and worsens on 20. Mel improves on 24 and worsens on 72; group MSE improves on 23 and worsens on 73. The endpoint waveform advantage is not consistent across all measured training steps: C's MAE was higher than B's at 250, 500, 1,000 and 1,500, including +8.195% at 1,500. This is one trajectory per initialization, not replicated evidence of a general advantage.

**Matched C2500 versus original2500.** These runs used the same first 30,000 ordered sources. The original reference is the earlier teacher-derived sliced initialization, not B.

| Metric | Original2500 | C2500 | Change |
|---|---:|---:|---:|
| Waveform MAE | 0.00230938 | 0.00237865 | +3.000% |
| Waveform MSE | 0.0000656531 | 0.0000677687 | +3.222% |
| Waveform NRMSE | 0.136074 | 0.138249 | +1.598% |
| Mel error | 0.148842 | 0.138365 | −7.039% |
| Group-output MSE | 0.00206382 | 0.00147844 | −28.364% |
| Group-output NRMSE | 0.129322 | 0.109456 | −15.362% |
| Active waveform correlation | 99.358093% | 99.349153% | −0.008940 pp |
| Quiet residual RMS, micro full scale | 81.0742 | 87.3101 | +7.692% |
| Quiet windows passing, of 2,544 | 1,123 | 872 | −251 |

Paired MAE improves on 23 sources and worsens on 73; correlation improves on 39 and worsens on 55. Mel improves on 90 and worsens on six; group MSE improves on 95 and worsens on one. Paired counts use strict numerical signs, without a significance tolerance. Error metrics cover 96 sources; correlation is the equal-source mean of the 94 sources with a defined active-sample cosine, not a pooled 96-source correlation. Exceeding 99% on this metric does not establish perceptual or silence acceptance. Final peak magnitude is 0.991052 with zero overshoot samples; peak is descriptive, not a score to minimize.

**Startup and quiet behavior.** C's 13 startup windows initially all passed, with residual RMS 1.955965 micro full scale. By the first measured review at update250, all 13 failed. They failed at every later review; the exact onset between updates 1 and 250 was not measured. At update250, 11 failed amplitude only and two failed both limits. At update2500, all 13 failed both residual and output-amplitude limits, with residual RMS 20.700613 micro full scale.

The following final groups partition the quiet panel. Near-silence means teacher RMS at most 1e-5; startup refers to the source-aligned first 20 ms. The seven named monitoring regions overlap and should not be summed.

| Disjoint final group | Windows | Passed | Residual-only failures | Amplitude-only failures | Both failures | Residual RMS, micro full scale |
|---|---:|---:|---:|---:|---:|---:|
| Startup near-silence | 13 | 0 | 0 | 0 | 13 | 20.700613 |
| Other near-silence | 171 | 152 | 0 | 19 | 0 | 1.484876 |
| Remaining quiet | 2,360 | 720 | 969 | 57 | 614 | 90.628145 |
| Total | 2,544 | 872 | 969 | 76 | 627 | 87.310114 |

The total RMS is sample-pooled, not an average of the three group RMS values. All 1,672 failures reconcile. The remote audit recomputed each saved gzip's complete seven-region summary at steps 0, 250, 500, 1,000, 1,500, 2,000 and 2,500 and verified unchanged window identities, teacher values and support. Local review independently checked all disjoint counts and paired-source totals.

C's final global MAE is 65.446% below its own initial value, mel is 40.999% lower, and quiet residual RMS is 41.152% lower. Those improvements coexist with the startup regression. From update1500 to2500, MAE falls 8.989% while quiet residual RMS rises 9.673%; quiet passes fluctuate rather than improve monotonically. Initialization constraints were not maintained as constraints during recovery. The observations establish loss of the initial startup behavior under this training policy, but do not isolate which parameter or loss branch caused it.

**Training onset coverage.** The CPU exposure audit counted 30,000 sources and 3,528,003,984 scored samples. A source-aligned first 20 ms was present in 14,696 crops; 1,866 had a near-silent teacher onset. Their 1,791,360 samples equal 37.32 seconds, or 0.050775% of the 20.416690 scored hours. Thus near-startup examples were present but occupied a small fraction of scored samples. This is exposure accounting, not a measured gradient contribution: mel, waveform and feature losses have different reductions and sensitivities. It neither proves a loss-weighting cause nor measures all near-silence outside startup.

**Integrity and reference limits.** The completion audit passed: all 30,000 teacher-cache comparisons, unique ordered source exposure, unchanged protected assets/files, finite group state, all 90 finite Adam moment states at counter2500, exact optimizer settings and parameter order, checkpoint identities, and fixed validation/window support. The saved final checkpoint SHA256 is `93c1b49bed08a8626f0289b4ca096f94df5cac588fdde7f7488a365f065ad5e0`.

The original5000 aggregate is authenticated against its protected checkpoint receipt. Its detailed development report is not bound by a protected report hash, so its sourcewise/detail comparisons are explicitly unbound and are not used for conclusions here. As a longer-training aggregate reference only, C2500 has 21.130% higher MAE and 24.262% higher quiet RMS than original5000; source exposure differs. The full remote aggregate-audit SHA256 is `852985c4ca9bda738becc9ae4fb3fadbcf113a280790f6e1a43184d163ff3064`; the local compact view omits overlapping region tables.

The completed run and all earlier candidates remain preserved. Future experiments start independently from the original teacher with a fresh optimizer and a 2,000-update budget. No new training run was launched during this completion review. The separate selector, hidden-map, startup-preservation and standalone constrained-A experiments are tracked in the experiment register.
