# Audit of the teacher-directed head objective

This is an analysis of saved results, with no additional model runs. Sources are [target-v2-summary.json](target-v2-summary.json), [wn-v2-summary.json](wn-v2-summary.json), and [short-v2-summary.json](short-v2-summary.json). Each arm starts from the same retained joint-spectral head and uses 256 updates over the same 2,048 training sources. The original checkpoint at step 8,890 is a separate reference. The canonical panel is reused development data, not an unseen final test.

## What the first comparison establishes

Teacher waveform supervision outside quiet regions is the appropriate eventual target. However, the particular weighting tested here gives it very little initial acoustic-gradient influence relative to quiet waveform error. Removing the old-student preservation term therefore removes a strong restraint without establishing balanced waveform guidance.

The comparison supports that concern, but it does not identify a complete causal explanation for every regressing recording or prove a defect in the decoder architecture.

| Change relative to the retained joint-spectral candidate | Repair control | Teacher-directed waveform |
|---|---:|---:|
| Natural waveform MAE | +0.128785% | +0.723993% |
| Natural mean mel error, equal crop weighting | −0.708218% | −1.187898% |
| Natural quiet residual RMS | −0.609196% | −0.944437% |
| Stationary silence residual RMS | −9.877792% | −6.861149% |
| Whole-source mel regressions above 1%, historical reduction | 2 | 21 |
| Natural sources with a quiet, active, or transition mel regression above 1% | 10 | 63 |
| Maximum waveform peak | 1.290138960 | 1.305236816 |
| Scored overshoot observations | 584 | 572 |

The retained candidate's maximum peak is 1.292609215 with 589 overshoot observations. A lower count does not establish improved peak behavior: the teacher-directed arm produces fewer above-limit sample observations but a larger maximum and regressions in several individual crops. Overlapping crops remain observations, not distinct physical events.

The repair control uses the common newly calibrated spectral coefficient. It is a matched control for this comparison, not an exact continuation of the previous selective experiment.

## The average spectral gain hides losses in quiet and transitions

These regional figures pool time-frequency elements within each resolution, then average resolutions. They differ intentionally from the equal-crop mel summary above.

| Regional mel-error change versus retained candidate | Repair control | Teacher-directed waveform |
|---|---:|---:|
| All natural spectral frames | −0.764176% | −1.325905% |
| Entirely quiet frames | −0.716998% | **+0.948166%** |
| Entirely active frames | −0.788371% | −1.559913% |
| Mixed quiet/active transition frames | −0.336787% | **+1.048587%** |

Teacher-directed fitting improves average spectra mainly through active audio. It worsens the pooled quiet spectrum even while quiet waveform RMS error decreases. Quiet waveform error and weak spectral detail therefore remain distinct criteria.

The teacher arm has 39 quiet-region, 19 active-region, and 32 transition-region regressions above 1%. These 90 region observations belong to 63 sources; they are not 90 independent recordings. The control has eight quiet and two active regressions, with none in transitions. This is not solely a sparse-window effect: 27 teacher-arm quiet failures and 27 transition failures have at least ten observed frames at every FFT size.

Examples include the existing Bodo and Manipuri active-speech failures, plus additional Hindi, Marathi and emotional recordings. Some of the largest relative quiet changes still have sparse support, such as four frames at only the smallest resolution. Such examples must retain their support counts. The natural quiet spectral analysis covers 79 sources, compared with 87 containing waveform-scored quiet samples.

On the independent 256-source selection split, final nonquiet displacement from the retained candidate is:

- Repair control MSE: `1.5129109641750816e-7`.
- Teacher-directed MSE: `4.000849444602201e-6`.
- Ratio: **26.4447 times in MSE, or 5.14244 times in RMS**.

This is consistent with the removed preservation restraint. It is not, by itself, proof that movement away from the old student is undesirable. Moving toward a better teacher is the goal; the relevant failure is that measured teacher-relative local quality does not consistently improve.

## Exact waveform-weighting calculation

The waveform objective is the sum of separately normalized region errors:

`L_wave = MSE_quiet / E_quiet,0 + MSE_active / E_active,0`.

The denominators are fixed measurements of the retained candidate on the training pool. They are not updated during fitting:

| Training-pool quantity | Quiet | Active |
|---|---:|---:|
| Baseline teacher-error MSE | `7.058194628679884e-8` | `0.0011153618418685701` |
| Baseline teacher-error RMS | `0.000265672629916593` | `0.0333970334291621` |
| Scored samples | 32,041,920 | 208,731,057 |
| Scored seconds | 667.54 | 4,348.5636875 |

The active-to-quiet error-scale ratio is **15,802.3673**. The sample-count ratio is **6.51431**. A normalized baseline loss near one in each region does not imply comparable gradients.

For a region containing `N_r` valid samples, error MSE `MSE_r`, and fixed denominator `E_r,0`, the waveform-output gradient norm is:

`||g_r|| = 2 sqrt(MSE_r / N_r) / E_r,0`.

At the full training-pool baseline, this implies a quiet-to-active gradient norm ratio of approximately **320.845**. The fixed 32-source diagnostic panel has different region counts and errors, so its measured ratio differs.

At the identical initial model, the saved output-gradient energies are:

- Repair-control waveform: `210.5093925134026`.
- Teacher-directed waveform: `210.5095963589228`.

The preservation derivative is zero at initialization. Quiet and active waveform derivatives occupy disjoint sample masks, so subtracting these energies isolates the newly added active teacher gradient:

`E_active = 210.5095963589228 - 210.5093925134026 = 0.000203845520218238`.

Its norm is **0.014277447959**, versus **14.5089418123** for quiet waveform error, a ratio of **1,016.21395**. Active waveform gradient energy is approximately **0.0000968343%** of total waveform gradient energy on this panel.

This is a measured initial **waveform-output gradient imbalance**. It must not be presented as a 1,016-fold ratio of parameter changes or AdamW updates. The head Jacobian, clipping, optimizer moments and coordinate scaling intervene afterward. Nor does this subtraction isolate the active gradient at the end, when preservation gradients are no longer zero.

## What the parameter-gradient probes do and do not prove

The new long-mel coefficient is `7.445405682710543`. On the fixed training-only panel, the teacher arm's waveform/mel output-gradient cosine is approximately +0.04118 initially and +0.04015 finally. After projection through the head into parameter gradients, their cosine is **−0.487989 initially and −0.536249 finally**.

At the end, weighted parameter-gradient norms are 9.18505 for waveform and 8.40217 for mel. The negative cosine establishes partial competition in this parameterization, not mathematical incompatibility. Their combined gradient still has positive inner products with both component gradients, approximately 42.98 and 29.21 respectively. An infinitesimal plain-SGD step along that combined negative gradient would locally improve both losses on this fixed panel. The saved probes do not prove that every actual AdamW update does so, or explain generalization to each development recording.

## Weight normalization and short-window results

Both follow-up arms retain the same initial candidate, data order, objective calibration and bounded budget. Their effects should be compared with the teacher-directed arm, not with the original checkpoint alone.

| Change relative to the teacher-directed arm | Weight normalization | Additional short spectral branch |
|---|---:|---:|
| Natural waveform MAE | −0.000209% | +0.096453% |
| Natural mean mel error | +0.001411% | −0.088583% |
| Natural quiet residual RMS | +0.003029% | +0.336960% |
| Stationary residual RMS | −0.002716% | +1.845672% |
| Maximum peak | +0.008266% | −0.067348% |

Weight normalization is effectively indistinguishable in useful quality terms within this pilot. It retains the same 21 whole-source failures relative to the candidate and seven relative to the original checkpoint. There is no observed complementary benefit that warrants combining it merely because it is available. This is not a general verdict against weight normalization in fresh or longer training.

The short-window branch has a modest, mixed signal. Relative to the retained candidate, transition-region failures fall from 32 to 18 and quiet-region failures from 39 to 28. However, aggregate quiet spectral error is slightly worse than in the teacher arm, natural quiet waveform improvement is weaker, stationary silence is worse, and whole-source failures rise from 21 to 22. Relative to original checkpoint 8,890, whole-source failures fall from seven to six. These are tradeoffs, not qualification. The short branch's parameter gradient is strongly aligned with long mel, with cosine approximately +0.752 initially and +0.750 finally. It is not an independent mechanism that can be assumed to cancel the other branch's failures.

## Principled corrective candidate and limits

A single controlled correction is a globally sample-pooled teacher waveform MSE with one fixed training-corpus teacher RMS scale:

`L_wave = sum_valid((prediction - teacher)^2) / (N_valid * s_reference^2)`.

This removes the separate inverse-baseline-error weighting. Keep the existing long spectral objective and recalibrate its coefficient using training-only gradients. Keep the same data, initial checkpoint, architecture and optimizer for the comparison. A global MSE may underemphasize very small quiet errors, so it is a testable candidate, not a promised silence solution. Quiet, local spectral and peak checks must remain in place.

The already scheduled WN, short-window, final-block and auxiliary-feature pilots remain useful observations under their recorded objective. Their failure would not establish that those methods fail under a corrected waveform objective.

Final qualification requires comparison against original checkpoint 8,890 as well as the retained experimental candidate. The disposable v2 calibration allowance of up to 1% peak movement is not a relaxation of final peak checks. Current automated source spectral gates cover the whole source; quiet/active/transition regions remain separately reviewed diagnostics. No saved result here establishes perceptual equivalence, production readiness or a new measured CPU RTF.

## Empirical follow-up: pooled waveform correction

The bounded corrective trial is complete. Evidence is `pooled-v1-summary.json`, with remote summary SHA256 `2c922e924e035c93a99f267397eb0aa4f765f180de17255dd40c63ab5a5d3720`. It used the same retained candidate, 2,048 distinct FIT sources in the same order, 256 updates, batch size eight and AdamW learning rate `1e-6`. The decoder architecture and deployed operations were unchanged. This trial corrects the separate region normalization; it does not establish a finished training recipe.

The fixed teacher scale was `0.0066966872767303025`, equivalent to RMS `0.08183328953873419`, measured over 240,772,977 valid FIT samples. `run_teacher_pooled_v1.py:144-175` computes the scale from teacher targets and divides the combined quiet-plus-active error sum by the combined valid sample count. Relabeling a valid sample quiet or active cannot change this waveform objective.

### The regional gradient inflation is removed

The first four calibration batches now have summed waveform-output gradient energies:

- Quiet: `3.370681390336787e-9`.
- Active: `7.679042475178672e-5`.
- Quiet share: **0.00438926%**; active-to-quiet gradient norm ratio: **150.93664**.

These are the actual gradients of global waveform MSE, not another balancing weight. Low-amplitude quiet errors naturally contribute very little squared error. The correction removes the earlier artificial inflation, but does not ensure enough optimization pressure to improve silence. This is consistent with the observed silence regression, without proving that gradient allocation alone caused every failed recording.

### Balancing output gradients did not balance head-parameter gradients

The recalibrated long-mel coefficient is `0.0010375157947540284`. Weighted waveform and mel output-gradient norms are close on the fixed training-only probe, but their parameter-gradient norms are not:

| Fixed-panel measurement | Before fitting | After 256 updates |
|---|---:|---:|
| Weighted waveform output-gradient norm | 0.00215124 | 0.00214792 |
| Weighted mel output-gradient norm | 0.00225654 | 0.00224092 |
| Weighted waveform parameter-gradient norm | 0.01825626 | 0.01648048 |
| Weighted mel parameter-gradient norm | 0.00118046 | 0.00142057 |
| Waveform/mel parameter-norm ratio | **15.4653** | **11.6013** |
| Parameter-gradient cosine | −0.159343 | −0.130295 |

Let `g_w` and `g_m` be the weighted parameter gradients. Then `g_m · (g_w + g_m) = ||g_m||² + ||g_w|| ||g_m|| cos(theta)`. Its measured value is **`-2.04048266e-6` initially and `-1.03241357e-6` finally**. Thus an infinitesimal plain-SGD step along the combined negative gradient would locally *increase* the mel objective on this fixed panel while decreasing waveform error. This gives a concrete explanation for why equal output-gradient norms are not a sufficient calibration contract after the head Jacobian. It agrees with the direction of the observed average tradeoff. It is not a reconstruction of actual AdamW updates: optimizer moments and coordinate scaling intervene, and only beginning/end probes were saved.

### Measured quality and remaining local failures

| Canonical metric | Retained candidate | Pooled trial | Change |
|---|---:|---:|---:|
| Waveform MAE | 0.0116127934 | 0.0115841882 | −0.246325% |
| Crop-mean mel error | 1.01888956 | 1.02516749 | +0.616155% |
| Natural quiet residual RMS | 0.0002532622 | 0.0002542213 | +0.378715% |
| Stationary residual RMS | 0.0000222316 | 0.0000259144 | +16.565821% |
| Maximum absolute peak | 1.29260921 | 1.29064643 | Lower |
| Overshoot sample observations | 589 | 568 | 21 fewer |

The lower maximum peak and total overshoot count do not satisfy the per-crop gate. Five crop observations still regress: two overlapping `freesound:25794` crops have 63→65 and 58→60 overshoot observations despite lower peaks; Kannada, Punjabi and Konkani crops have higher peaks. This represents four sources, not five independent recordings.

Local spectral regressions exceed the 1% guard in 56 legacy whole-source metrics, or 52 with the pooled spectral aggregation. By region, the pooled comparisons contain 53 active, 21 quiet and 14 transition failures, besides the 52 whole-source failures. The union contains 69 sources; region and whole-source counts overlap. All pooled regional mel averages worsen: quiet +0.612743%, active +0.772125%, transition +0.416592%. The largest quiet regressions include Japanese emotional sources with 164/63/18 and 202/85/27 frames across resolutions, so this is not exclusively a sparse-frame artifact.

Against original checkpoint 8,890, the trial still preserves useful inherited improvements: MAE −0.162361%, mel −2.191860%, natural quiet RMS −1.653277%, and stationary RMS −40.825445%, with no legacy whole-source mel regressions above 1%. However, four crop peak regressions remain against that original reference. The existing `freesound:220655` quiet spectral failure also remains, at +6.242364%, supported by only 8/2/0 frames. Its sparse support limits that single estimate; it does not excuse the independently supported failures against the retained candidate.

No selection checkpoint at steps 64, 128, 192 or 256 qualified. Quiet RMS on that independent 256-source selection split worsened by approximately 0.193%, 0.237%, 0.311% and 0.406%, respectively. Nothing was promoted.

The empirical conclusion is narrower than a recommendation to use global pooling: it corrected one verified weighting defect and exposed a second calibration limitation, while trading some retained spectral and silence quality for waveform error. Keep this as a diagnostic result. It does not justify combining the candidate with other arms, assuming further training will fix the tradeoff, or launching an open-ended series of new trials. The already authorized late-block and auxiliary-feature pilots should be interpreted under their own recorded objectives and matched controls.
