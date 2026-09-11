# Startup retention and the remaining pruning experiments

11 September 2026. The current model is the original AudioVAE2 decoder with a narrower stage, not the earlier ConvNeXt student. This experiment keeps the 384/256/128 group widths and all nine residual units. Teacher, encoder and outer decoder stages remain frozen.

## Recommendation

Keep the GRAIL-like initializer as the strongest matched 2,000-update waveform candidate. Establish useful startup-preserving recovery separately before combining them. GRAIL is a folded initialization method here, not a different inference architecture and not a guarantee that later training retains its fitted behavior.

| Matched 2,000-update result | B native reconstruction | G shared hidden correction |
|---|---:|---:|
| Waveform MAE, lower is better | 0.00261465 | **0.00247640** |
| Active waveform cosine | 0.992375 | **0.992986** |
| Quiet windows passing | 1,242/2,544 | **1,419/2,544** |
| Startup windows passing | 0/13 | 0/13 |

G reduces waveform MAE by 5.29% relative to B, with improvements on 80 of 96 development recordings. Internal group MSE and the teacher's 20–40ms transient do not improve uniformly. Cosine is not perceptual accuracy, and these measurements do not establish CPU RTF. [Completed G comparison](../grail-hidden-v1/completed-findings.md).

## What is causing loss of startup behavior

The same pruned architecture already passes all 13 measured startup windows under the constrained C initialization. Therefore, failure during subsequent recovery is not proof that another inference layer is required. A fresh diagnostic reproduces the loss after just four ordinary updates, with all teacher targets and execution paths qualified.

The failure is specifically an increase in near-silent output level: 13/13 passes become 2/13, while all residual-error bounds still pass. Output RMS rises from 9.406 to 10.642 microFS; teacher RMS is 9.334 microFS. Both DC and AC components increase. This is not full-scale clipping and is not solely a bias error.

Eight parameter interventions show why protecting the stage-3 upsampler alone is insufficient. Moving that operator alone retains 13/13 passes. Moving upstream and downstream group parameters while keeping that operator at its initial weights gives only 2/13. The fitted operator depends on features that joint recovery changes, and subsequent trainable blocks also change their response. This is a measured interaction on the early trajectory, not proof that either block should remain frozen for general reconstruction. [Layer interventions](measured-findings.md).

The original global waveform, mel and group-feature objective does not require every rare startup window to remain inside its bounds. In the earlier C run, near-silent first-20ms onsets comprised 0.050775% of scored waveform samples; this is sample exposure, not a measured gradient share. A frozen teacher defines the target, but does not impose per-window invariance during optimization.

## What theory suggests

A folded fit such as `W X = Y_teacher` is fitted at particular hidden features. When upstream recovery changes X, the original fit is no longer sufficient to preserve Y. This limitation is consistent with the distribution-shift limitations described in [GRAIL](https://arxiv.org/html/2602.23795v1#S3).

At an almost exact waveform match, a squared residual has gradient `2 J^T error / n`, which can be tiny while the output Jacobian J remains substantial. A finite step can still introduce the positive term `||J delta||^2 / n`. [Orthogonal Gradient Descent](https://proceedings.mlr.press/v108/farajtabar20a.html) motivates protecting outputs instead of inferring insensitivity from a near-zero loss gradient. Our measured early update exhibits a wrong-sign finite residual prediction, but this does not uniquely separate nonlinear curvature from all network arithmetic.

The practical transfer is to keep a small declared set of training-calibration startup inputs involved in recovery, refresh their current output constraints, and check the real waveform after each proposed update. [GEM](https://papers.nips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) and [FROMP](https://papers.nips.cc/paper_files/paper/2020/file/2f3bbb9730639e9ea48f309d9a79ff01-Paper.pdf) provide related memory/functional-preservation ideas. Neither paper proves our exact hard constraints or codec performance.

Six recurring calibration anchors are training involvement, counted separately from unique ordinary sources. The 13 development starts remain evaluation-only. Passing all six cannot guarantee all thirteen, and our intervention and recovery results directly demonstrate this limitation.

## Completed retention pilot and diagnostic

Two independent fresh-C pilots used the same original RNG, empty Adam and 768 distinct ordinary sources, each once. The retention arm additionally protected six fixed calibration starts.

| After 64 updates | Ordinary recovery | Startup retention |
|---|---:|---:|
| Development startup passing | 1/13 | **12/13** |
| Calibration startup passing | 0/6 | **6/6** |
| Waveform MAE | **0.00488615** | 0.00542730 |
| All quiet windows passing | 413/2,544 | 456/2,544 |
| Rejected weight updates in final 16 | 0 | **16** |

The retention arm initially learned, reducing waveform MAE 21.16% from initialization. It then stopped moving despite advancing its declared Adam moments. Extending that implementation to 2,000 updates was therefore deferred. [Full pilot comparison](pilot-findings.md).

The targeted replay reproduced the first rejected update at step 43, matching every recorded non-timing update scalar. It found an avoidable grid cutoff: the smallest tested 1/16 displacement failed, whereas 1/32 passed all six constraints with real parameter changes. All twelve individual linear predictions already passed at 1/16, so linear constraints alone did not predict the real waveform. Same-prediction FP64 scoring retained the failure, separating it from a final metric-reduction flip. Larger proposals also violated omitted linear rows. [First-zero evidence](first-zero-findings.md).

The isolated extended-grid pilot has now completed, matching the old non-timing scalar prefix through update 42. It accepts a real 1/32 update at step 43, but still rejects 15 of the last 16 updates. Waveform MAE improves only 0.148% over the original retention pilot, and development startup passing falls to 11/13. All six calibration anchors remain passing. Thus the grid fix addresses the first avoidable rejection without establishing useful sustained recovery. [Completed grid comparison](grid-pilot-findings.md).

Both full 2,000-update retention runs remain deferred. The complete twelve-row direction comparison and its separate fresh 64-update pilot have now completed. The pilot improves MAE 7.09% versus the extended grid, with 12/13 development starts and 6/6 calibration starts passing. It makes nonzero changes in twelve of the last sixteen updates, but mean accepted fraction falls below 0.7% in the final eight. The next bounded test replays this policy's first rejected update at step 56 and compares a current-trial normal correction. [Complete pilot findings](complete-pilot-findings.md).

Related nonlinear optimization methods use higher-order or normal corrections, but their convergence claims do not transfer automatically to Adam and this decoder. The new correction remains a separate local test, not a relabeling of the all-row pilot. [Primary-method review and conditional repair](feasible-update-review.md).

## Remaining experiments

| Item | Disposition |
|---|---|
| Downstream-aware selection plus B | Completed at 2,000 updates. Preserve its results; no immediate repeat. |
| GRAIL-like hidden correction | Completed at 2,000 updates. Retain as the promising overall waveform result, with limitations. |
| Startup-preserving recovery | Numerical and causal diagnostics plus matched 64-update pilots completed. Qualify the update rule before full recovery. |
| Extended-grid retention | Completed fresh 64-update pilot; old scalar prefix through update 42 authenticated. Still stalls, so no full-run extension. |
| Complete individual constraints | Same-state direction comparison and fresh 64-update pilot completed. Best measured retention pilot so far, but late movement remains restricted. Preserve code and results. |
| Bounded nonlinear correction | Next local test at the all-row policy's first zero, update 56. Exact fresh-state replay precedes disposable candidate comparison. |
| G plus startup retention | After retention works, declare a fresh-teacher-derived G initialization that itself passes the training startup constraints. Do not start from trained G or silently accept its existing startup failures. |
| Original current-batch-only Q | Superseded as the next trial because startup is unprotected in batches without startup. Preserve its failed evidence and code. |
| Standalone constrained-A recovery and further width cuts | Deferred while the measured retention problem is unresolved. |

All prior checkpoints, sources and results remain preserved. New pilots are disposable and export only aggregate statistics. No full recovery, deployment promotion or extra compression cut is implied by this report. The chronological [experiment register](../experiment-register.md) preserves earlier methods, failures and completed budgets.

## Combining GRAIL and startup recovery without another layer

No new inference layer is justified by these results yet. Initial startup feasibility is demonstrated in the existing pruned architecture; sustained feasibility during useful ordinary recovery is unresolved. GRAIL here changes surviving weight initialization. Startup retention changes the training update policy. They can therefore be combined in one decoder without separate inference branches.

The proposed combined experiment starts fresh from the original teacher, applies the declared GRAIL-like compensation, then performs a separately declared startup calibration to obtain a feasible starting point. G itself is not initially startup-feasible, so this preparation cannot be skipped. Recovery keeps the frozen encoder, original latents, teacher waveform targets and ordinary losses, while applying startup-specific supervision or a qualified preservation rule on declared training inputs. All shared group weights remain trainable.

The preservation rule must first permit useful nonzero learning. Do not splice trained retention-pilot weights into G, freeze only the upsampler, or assume that weights assigned to startup cannot affect ordinary speech. The measured upstream/downstream interaction makes that separation unsafe to assume. Complete-constraint and, if required, nonlinear-correction tests should determine a viable update before a fresh combined 2,000-update comparison. No architecture change is ruled out permanently, but the current evidence favors fixing training before adding inference cost.

## Current result: useful startup retention without extra inference layers

The bounded normal-correction diagnostic and a fresh 64-update pilot are now complete. The pilot makes real weight changes on all64updates, with28 accepted corrections and no rejected updates or grid fallbacks. All six calibration starts stay passing. Development ends at12/13; ordinary waveform MAE matches the ordinary-control pilot within0.004%. This is useful retention qualification, not complete silence recovery. Ordinary quiet passing is466/2544 and the20–40ms transient cohort remains0/10. [Completed pilot](corrected-pilot-findings.md).

A fresh G-plus-constrained-upsampler initializer has passed under the [combination plan](fresh-grail-combination-plan.md): six calibration and thirteen development starts pass before recovery. The user has approved the subsequent 2,000-update run after matched short qualification, and requested a documented reusable procedure for later cuts. Further block removals or 50% channel cuts are not included in this current geometry. All earlier sources and failed attempts remain preserved.
