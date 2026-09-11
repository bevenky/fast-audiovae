# Two independent follow-on experiments

Status: authorized independent protocols. Implementations and tests are being finalized before each remote launch. Keep the downstream-selector experiment, previous B and combined runs, original teacher, data ledger and all previous artifacts. Neither arm automatically extends beyond 2,000 updates or changes the next progressive cut.

Both arms use width384/256, all90 trainable group tensors, the original teacher/encoder and causal geometry, original RNG, fresh AdamW3e-5, the existing betas/epsilon/weight decay, physical singleton execution, accumulation12, and exactly the original first24,000 distinct recovery sources in their original order. Preserve the existing raw waveform L1, mel and full-group MSE coefficients. Evaluate the same96 development recordings and unchanged seven quiet cohorts at0/250/500/1,000/1,500/2,000. Development never chooses coefficients, channel sets, constraint targets or training samples. Report calibration cost, training cost and quality separately.

## Arm G: shared hidden reconstruction folded into native weights

Question: does a constrained, shared hidden-feature reconstruction map initialize recovery better than B's unrestricted native output-error fit? Use the original pivoted channel set, not the new downstream selector. This changes only the compensation policy relative to B2k. Do not add startup equality constraints or any new recovery loss.

GRAIL learns a linear map from reduced post-activation features to original hidden features and folds it into the consumer weights; its convolutional formulation shares the channel map across kernel taps. It explicitly avoids folding through a nonlinearity. Our causal, sequential, actual-student-input adaptation is **GRAIL-like**, not a replication of its vision/LLM experiments or a published codec recipe. [GRAIL, section3](https://arxiv.org/html/2602.23795v1#S3), [official implementation](https://github.com/TWWinde/GRAIL_Compensation).

Start a fresh384/256 slice from the original teacher. Process RU1, RU2, RU3, then stage3 upsampler. At each site, recompute its post-Snake consumer input on the current student, after applying earlier corrections. Collect the corresponding complete original teacher input, independently from the frozen teacher. If X is the student's384-channel input and H the teacher's512-channel input, fit

`R = argmin_R sum_t q_t ||H_t - R X_t||² + lambda ||R||F²`

`R = (H Q Xᵀ) (X Q Xᵀ + lambda I)^(-1)`

`lambda = 1e-6 * trace(X Q Xᵀ) / 384`.

Use FP64 uncentered moments and a Cholesky solve, with no fitted mean or intercept. The factor1e-6 is our fixed existing numerical policy, not the paper's reported regularization range. Record conditioning, actual residuals and solve/writeback parity. If the input Gram has zero trace or nonfinite entries, stop with a numerical failure rather than choose a new ridge from development results.

For a residual pointwise consumer, form `Wnew = Wteacher[S,:] R` and retain `bias_teacher[S]`. H includes all512 teacher input coordinates. Selected rows of R are not forced to identity; the reduced student input may already have drifted. The residual skip remains the actual current student skip. Consequently this arm does **not** explicitly compensate current skip error as B does with its `teacher_RU[S] - current_skip` target. The held-out full residual/group/waveform measurements must expose any cost of that difference.

For the stage3 upsampler, use the **same** R at every one of its ten native taps:

`Wnew[k,o,tap] = sum_c Wteacher[c,o,tap] R[c,k]`.

Retain the teacher's single256-element bias. The map acts on post-Snake inputs only; do not merge it through Snake or the residual identity. Do not add an affine feature offset: across five phases and missing startup history, its tap-dependent constant generally cannot be represented by the existing one shared output bias.

Pointwise feature-cell weight q is the existing count of valid waveform samples, at most40. For upsampler input cell t, q is the sum of valid waveform counts in the five current-phase outputs at t **and** the five outputs at t+1 that consume it as previous input. Keep actual context cells that feed scored output and exclude cells that feed none. This uses one shared map for both taps, without resetting history at the scored start or training on padded tails. It is a hidden-reconstruction usage weight, not the native output-error objective used by B.

After each fit, write only existing WN effective weight coefficients and preserve the native bias exactly. Recompute actual inputs for the next fit. Save the four maps as analysis sidecars, but export only the four changed native operators plus authenticated selection and lineage. There are no added inference modules, offsets, filters or phase terms. Comparing hidden-map error alone is insufficient; full96 development inference must follow native FP32 folding.

Cost is four72-source calibration passes, four384-dimensional solves, one96-source step0 evaluation and the2,000-update recovery. This is smaller algebra than B's3,840-dimensional upsampler fit, although the forward/runtime cost is unmeasured. Required tiny tests cover all ten tap folds, actual zero-padding startup, partial/context-only masks, asymmetric actual inputs, uncentered fitting, unchanged bias, WN writeback and source/state isolation. The whole candidate starts afresh from the teacher, never from an adapted B/combined/old5000 model.

## Arm Q: quiet-aware projection of actual Adam displacements

Question: can recovery retain the feasible combined initialization's quiet behavior while continuing ordinary reconstruction learning? Start from the **exact authenticated combined step0 initializer**, freshly reconstructed from the original teacher. Its comparator is the saved ordinary combined trajectory through2,000, with the identical initializer, source order, RNG, optimizer and losses. Do not compare this as a single-variable test against B, which starts with a different initializer. Do not add the new selector or GRAIL maps.

A projected-update construction is principled without inventing a quiet loss multiplier. GEM is a precedent for a small quadratic projection protecting auxiliary loss inequalities; it uses remembered examples and gradient directions. The method below instead protects only naturally present current-batch quiet windows and projects the **actual Adam displacement**, so it is not GEM replication or a continual-retention guarantee. [GEM, section3](https://papers.nips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf).

Use unchanged teacher-defined20ms windows, scored spans, source offsets and quiet limits. Partition current-batch quiet windows into three disjoint sets: teacher-near startup at absolute source0–20ms; other teacher-near windows; remaining ordinary quiet windows. Teacher-near means RMS<=1e-5; ordinary quiet retains the evaluator's existing quiet cutoff. This separation prevents a large ordinary-quiet error from hiding a near-startup constraint.

For each present set s, define two differentiable-at-a-unique-maximum statistics, using physical squared full-scale units rather than inverse quiet-RMS weighting:

`F[s,res](theta) = max_w( mean((student_w-teacher_w)^2) - residual_limit_w^2 )`

`F[s,amp](theta) = max_w( mean(student_w^2) - output_rms_limit_w^2 )`.

Limits are exactly those of the existing evaluator. No learned teacher threshold, gate relaxation, or new target appears. At exact ties select the first authenticated source/window deterministically. At the pre-update state define each cap `c_j=max(0,F_j(theta))`. Thus a currently feasible cohort must remain feasible in this batch; an infeasible cohort's worst physical excess must not increase. This is deliberately weaker than forcing all failing windows to pass in one update and does not guarantee improvement of every individual failing window.

For numerical agreement, implement each squared-energy term through the existing evaluator's FP32 `vector_norm / sqrt(valid_count)` RMS, then cast that RMS to FP64 before squaring and subtracting the unchanged limit squared. This is mathematically the MSE expression above but follows the deployed metric's rounding order. Nonlinear cap comparisons are exact in that defined arithmetic; no quality tolerance is added.

Accumulate the unchanged ordinary objective across12 new sources and all90 group tensors. Compute at most six constraint gradients at the same pre-update state, independently of ordinary `.grad` storage. Obtain the actual FP32 Adam proposal d0, including its current moments, bias correction and epsilon, by advancing the existing Adam state exactly once. Save that proposed moment state and restore parameter values before applying any correction. Never project raw gradients and assume Adam preserves the resulting inequalities.

With `a_j=gradient(F_j)` solve

`min_d 0.5 ||d-d0||² subject to a_j dot d <= c_j-F_j(theta)`.

All90 parameter tensors participate in a fixed authenticated order. This is a Euclidean projection in the model's existing WN parameterization, not a claim of parameterization invariance. Normalize each nonzero constraint row and its right side together for conditioning; this leaves the feasible set unchanged. A zero row imposes no first-order restriction, but is still checked after the candidate update. The zero displacement is always feasible because all right sides are nonnegative.

The dual has at most six variables: `min_lambda>=0 0.5 lambdaᵀ(A Aᵀ)lambda - lambdaᵀ(A d0-b)`, with `d=d0-Aᵀlambda`. Compute its small Gram/dot products in FP64, handle duplicate/linearly dependent rows explicitly, and verify primal feasibility, dual signs and complementary slackness. This requires no large Hessian or full waveform Jacobian.

The neural constraints are nonlinear and a maximum can switch windows. Therefore test the actual native student outputs after candidate parameter updates on the **same current batch**, reusing detached cached teacher targets. Try displacement fractions1,1/2,1/4,1/8,1/16 in that fixed order, then0 if none meets every nonlinear F_j cap. Recheck complete current-batch cohorts, not only the windows that supplied gradient rows. No tolerance may change a reported quiet pass into a pass; the existing deterministic metric arithmetic must reproduce the pre-state at fraction0 or fail the numerical check. Record all zero-displacement events and maximum-switch events. Do not silently turn a numerical QP failure into an ordinary unconstrained update.

Adam moments and counters advance once per unique12-source batch from the ordinary gradient; constraint gradients are not injected into those moments. A rejected displacement leaves weights unchanged but does not replay the sources or reset the moment state. Thus2,000 means2,000 processed optimizer batches, with accepted nonzero displacement count reported separately. This is an explicit projected-Adam policy, not identical weight dynamics to ordinary Adam.

Forward/gradient recomputations inside a single update are numerical evaluation of that update, not reuse in a later training batch. Do not append old calibration/dev/startup recordings to the stream, oversample sources or create artificial zero-context crops. Cache no future labels. Most importantly, **a batch lacking startup has no startup constraint; later updates can damage previously successful startup outputs**. A cohort whose pre-update worst excess is already positive may also permit individual previously passing windows to cross their limits while staying below that worst cap. Report these limits rather than call the policy a hard global preservation guarantee.

## Quiet-arm implementation and decision gates

Before launch, tests must prove: no-constraint behavior is bitwise ordinary Adam; projection uses actual displacement rather than raw gradients; all90 tensors and correct moment/counter advancement; known small QP solutions and dependent-row handling; deterministic max selection; nonlinear backtracking catches a changed maximizing window; zero-step restoration; teacher-window/crop identity; and unique source exposure. This arm needs more student computation: at most six extra constraint backpropagations plus five candidate scoring rounds on the current batch's quiet-containing sources. Reuse teacher intermediates, do not retain twelve large graphs, and report actual memory/time and intervention rates. No inference cost is added.

Quality comparisons retain ordinary MAE/mel/full-group error, active RMS, per-recording errors, startup/other-near/ordinary quiet residual and amplitude failures, all seven existing cohorts and peak invariants. New diagnostics include constraint availability by cohort, proposal conflicts, displacement correction norm/cosine, backtracking/zero-update counts, and before/after current-batch cap checks. Improvement restricted to constrained training windows is insufficient; the fixed96 held-out results decide usefulness. Do not promote an arm merely because quiet improves while ordinary audio or throughput deteriorates materially. A negative result tests this bounded current-batch protection policy, not whether silence preservation is impossible.

GRAIL implementation may proceed after its specified tests; the quiet projected-update policy has passed independent method review, with implementation validation still required before launch. Both runs remain independent and preserve all previous experiments.
