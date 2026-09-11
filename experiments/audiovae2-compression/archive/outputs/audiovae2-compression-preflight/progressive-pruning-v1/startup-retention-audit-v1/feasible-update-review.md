# Nonlinear feasibility after the 64-update retention pilot

11 September 2026. Saved-result analysis and primary-source review only. No new model execution, tests or training implementation. The first-zero-step diagnosis is pending; the mechanism and proposed remedy below remain conditional.

## What the completed pilots show

Both [ordinary](ordinary-pilot-v2-aggregate.json) and [startup-anchor](anchor-pilot-v2-aggregate.json) pilots completed 64 updates from the same fresh C state and passed preservation checks. R retains all six calibration starts and 12/13 development starts, versus 0/6 and 1/13 for ordinary recovery. R development MAE is 0.005427298 versus 0.004886148 for ordinary recovery, 11.075% worse; it nevertheless improves 21.159% over initialization. This is useful initial retention with a recovery cost, not a successful finished recipe.

R has 21 zero parameter displacements, including every update from 49 through 64. Its first zero occurs at update 43, with amplitude-excess slack −5.855374e-17. The proposal norm is 0.01748443, its linear correction norm is only 0.0000925972, and proposal/projection cosine is 0.99998598. A small correction to the parameter vector therefore does not establish a usable nonlinear step. Adam moments continued to advance on the zero-displacement updates, as declared.

The existing positive fraction grid is 1, 1/2, 1/4, 1/8 and 1/16. Rejection on this grid does **not** show that every positive step is infeasible. The saved final amplitude slack is −1.522392e-15, not exactly zero. Smaller representable steps, omitted nearly active windows, nonlinear curvature and FP32 rounding must be distinguished by the scoped diagnosis. There is no basis for a 2,000-update launch while the terminal stall remains unexplained.

## The relevant theory, and the mismatch with our implementation

For one smooth constraint F(theta) ≤ 0 and a direction d,

`F(theta + alpha*d) = F(theta) + alpha*g^T*d + alpha^2*d^T*H*d/2 + higher-order terms`.

At an exactly active boundary, a tangent direction with `g^T*d = 0` and positive directional curvature violates the constraint for every sufficiently small positive straight step. At a strictly negative but tiny slack, an admissible mathematical fraction may exist below the tested grid or below useful FP32 parameter resolution. Also, a two-maximum linear model need not protect another almost-maximal window with a different gradient.

This resembles **Maratos-type rejection at a curved constraint boundary**. The classical Maratos effect concerns poor step acceptance and lost fast convergence in nonlinear programming; the current routine is an Adam-displacement projection with strict physical checks, not an SQP algorithm satisfying those papers' hypotheses. The name is a useful analogy only after local constraint values and directions support it.

Three primary sources clarify the applicable mechanisms:

- **Hu, Xiao and Chen, feasible active-set SQP, section 2, Steps 5–6.** Their method obtains a higher-order correction from a reduced least-squares problem and searches a curved path `x + alpha*d + alpha^2*r`, checking every original inequality. Feasible-descent construction and regularity assumptions matter; merely importing the correction formula does not import its convergence result. [Original paper](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E29AA9E08DEF00432CA733DD2BF9A3B9/S0004972700035577a.pdf/an_active_set_sequential_quadratic_programming_algorithm_for_nonlinear_optimisation.pdf).
- **Wächter and Biegler, section 2.4, equations 24–28, and section 3.3.** IPOPT repairs a rejected proposal using its measured nonlinear constraint residual and a Newton-type correction, reusing the existing Jacobian/factorization where possible. Corrections are bounded in number; inadequate improvement returns to shorter steps. A separate restoration phase handles constraint violation. Its filter permits intermediate nonlinear infeasibility, so our adaptation must keep the exact all-window acceptance rule rather than copy the filter or its convergence claims. [Original implementation paper](https://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf).
- **Mao, Szmuk and Açıkmeşe, successive convexification, section II and Algorithm 1.** Their optimal-control algorithm repeatedly linearizes nonlinear dynamics, uses trust regions and compares predicted with actual improvement. Virtual controls can temporarily violate the original dynamics. This supports validating a local model before taking its full step, but not claiming that a convexified subproblem certifies our nonlinear waveform constraints. [Original paper](https://arxiv.org/pdf/1608.05133).

These are optimization results, not codec-startup experiments. None establishes a latency or recovery-quality improvement for this decoder.

## What the first-zero diagnosis should distinguish

Use all **12 individual inequalities**, residual and amplitude for each of the six calibration windows, with original bounds. Compare `F_i + g_i^T*d` to actual values along the same realized FP32 displacement. Record which omitted rows are linearly violated, maximizing-window switches, positive fractions and their actual parameter changes. The canonical evaluator remains the acceptance authority; an FP64 diagnostic is explanatory, not an alternative relaxed gate.

If a formerly omitted row is already linearly violated, complete the linear constraint set first. A two-row maximum selector was too small at that state; that finding would not require a curvature remedy. If all individual linear predictions pass but actual nonlinear values fail with a quadratic-scale discrepancy, curved-boundary rejection is supported. If only smaller fractions pass, the coarse grid also contributed. A small nonzero violation comparable to rounding needs numerical qualification before assigning it entirely to curvature.

## Smallest conditional repair to test

Keep the ordinary Adam proposal and its moments unchanged. The experimental operation is a bounded projection/repair of that proposal, followed by exact checks of every anchor. All original loss weights and thresholds stay fixed.

1. **Complete the relevant linear constraints.** At this six-window scale, inspect all 12 rows, or add every violated row through a verified working-set procedure. Dependent rows need rank-aware treatment; failure of a numerically singular solve is not proof that no feasible movement exists.
2. **If a linearly acceptable trial is nonlinearly rejected, compute a normal correction from its actual excess.** For trial `v = theta + d`, a minimal proposal is `min_r ||r||^2/2` subject to `F_i(v) + J_i*r ≤ 0` for all 12 rows. Reuse the base Jacobian for the first correction only if the measured local prediction is adequate; otherwise refresh it at v. This is a small dual problem even though the parameter vector is large. It needs no full Hessian and no new inference operation.
3. **Verify a genuine feasible corrected point.** Evaluate the actual model after correction with the original FP32 constraints. A bounded search along the normal correction, or a bounded refreshed correction, can seek an inward point; if it fails, reduce the tangential proposal and recompute. Never accept a residual positive excess, change a limit, or keep a partially repaired infeasible state. Restore the exact prior weights when no proposal passes.

A plain Newton correction to the boundary is insufficient even in a simple convex example. For `F(x,y)=x^2+y^2-1`, start at `(1,0)` and take tangent step `(0,t)`. The base-Jacobian correction `(-t^2/2,0)` leaves `F=t^4/4 > 0`. Thus a correction can dramatically reduce violation while still failing strict acceptance. A short normal-direction bracket, for example testing correction scales 1 and 2 before reducing the proposal, is a possible diagnostic of inward repair. Those are algorithmic search choices, not paper constants or changed quiet tolerances. Every corrected candidate still needs all-window verification, and normal overstepping can harm another constraint.

The normal correction must be bounded relative to the proposal and the number of rescans must be predeclared. A correction that becomes comparable to replacing the entire proposal, or repeatedly yields zero movement, is not established useful recovery. The first test should be the already identified stalled state, with no retained optimizer update, before considering another matched pilot. A successful finite repaired step would justify testing persistence, not a full-run guarantee.

## Output-linearized alternative if scalar corrections are inadequate

The physical constraints can also be represented by convex norm bounds on an affine waveform approximation. Let `p` be the current startup output and `Jp` its waveform Jacobian. Enforce

`||p + Jp*d - teacher||_2 ≤ sqrt(n)*residual_limit`

and

`||p + Jp*d||_2 ≤ sqrt(n)*output_limit`

for each window, while minimizing distance from Adam's proposal. This retains the nonnegative squared-Jacobian term that a first-order expansion of the scalar squared error discards, including when initial residual gradients are tiny. It still omits curvature of the neural waveform map and still requires true nonlinear checks. Jacobian-vector products and a small correction subspace could avoid materializing a full waveform-by-parameter Jacobian, but this is more machinery than a bounded scalar normal correction. It is a fallback proposal, not the first implementation recommendation or an executed test.

## Decision and limits

Hold full R training pending the first-zero diagnosis. If the issue is an omitted linear constraint, test complete-row projection first. If curved-boundary rejection is confirmed, test a bounded normal repair and all-window acceptance at that same state. Do not add current-batch quiet objectives, new startup loss weights, a quieter target, a changed threshold, a new architecture or another width cut to this diagnostic.

Retained finite-anchor feasibility must be accompanied by nonzero learning and ordinary quality. Six calibration constraints remain recurrent training involvement, separately counted from the unique ordinary source stream; the 13 development starts remain evaluation-only. No finite-anchor method guarantees all 13 will pass, and R64 already demonstrates that distinction. Completed D and G, all previous initializers and the failed original Q remain unchanged and registered.
