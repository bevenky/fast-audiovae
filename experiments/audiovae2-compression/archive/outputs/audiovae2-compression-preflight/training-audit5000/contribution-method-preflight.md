# Method preflight for the channel-contribution diagnosis

Scope: first stage-2 residual pointwise mixer, plus stage-3/stage-4 upsampler decomposition and teacher ablation. Use the original teacher and the authenticated sliced initialization. Do not modify or insert teacher-coordinate corrections into the adapted 5,000-step model. This document specifies a bounded diagnosis; it does not report new results.

## Distinct controls

1. **Teacher operator decomposition.** At the original teacher input to an affected linear operation, split selected outputs into retained-input contribution, discarded-input contribution and copied bias. For an upsampler, preserve the actual current/previous-input taps and phase routing. Confirm that the components reconstruct the recorded operator output within the declared numerical gate.
2. **Teacher-only ablation.** Remove only the discarded-input contribution to selected output channels of one teacher operation. Keep other output channels and all downstream teacher operators. This isolates that contribution in the teacher's own representation. It does not emulate the whole narrowed student, which lacks other channels and connections too.
3. **First-matching initialization restoration.** At stage-2 RU1, verify that sliced initialization retains matching features through the channelwise operations before its first pointwise mixer. Add the exact discarded contribution to that mixer's selected output, then run the unchanged downstream initial student. This is an oracle diagnostic using teacher information, not a deployable repair. Later omissions remain and can interact with the correction.

At later initial-student operators, use distinct selected output/input sets A/B and omitted input set D. In conceptual output-by-input matrix notation:

```text
teacher_y[A] - student_y
  = W[A,B] (teacher_x[B] - student_x) + W[A,D] teacher_x[D]
```

Bias cancels only because the initial sliced bias matches. Convolution/phase contractions replace matrix multiplication where appropriate. This distinguishes prior input drift from the direct lost contribution. The inputs are those actually entering the linear operation, after Snake and conditioning. Do not extend this identity to adapted weights without accounting for their changes.

## Fixed affine reconstruction probe

Fit **only the first mixer** initially. With `x=phi[kept]` and `d=W[kept,dropped] phi[dropped]`, fit `d≈A x+c` using the 72 source-disjoint calibration sources. Use every valid stage-2 cell. Weight a cell by its valid waveform-sample count, from 0 to 40, so partial tails and differing durations are pooled explicitly. No division by quiet RMS and no development-selected weights are allowed.

Use float64 centered weighted statistics, preferably stable centered accumulation/merging:

```text
Cxx = sum(w (x-mux)(x-mux)^T) / sum(w)
Cdx = sum(w (d-mud)(x-mux)^T) / sum(w)
ridge = 1e-6 * trace(Cxx) / 256
A = Cdx (Cxx + ridge I)^(-1)
c = mud - A mux
```

Solve the system rather than explicitly forming an inverse. The intercept is unpenalized. If the input variance is exactly zero, use a flagged intercept-only result rather than an undefined solve. Report numerical-rank tolerance, effective rank, eigenvalue/condition diagnostics before and after ridge, coefficient norms, training source/cell/sample counts and fitted-weight change. A single fixed ridge avoids choosing the solution on the development panel.

Fold the correction into the existing effective pointwise weight and bias: `Wnew=Wkeep+A`, `bnew=bkeep+c`. Rebuild weight-normalization parameters with the existing helper. There is no new activation, layer, runtime tensor or residual-skip change. The exported float32 folded calculation must be checked against its intended affine correction; float64 fit quality alone does not validate the folded model.

This fit is not a complete reconstruction-aware initializer for all stages. For a later residual-unit fit intended to match the whole unit, the branch target would need to include skip mismatch, `teacher_unit_output[kept] - student_skip`. Merely fitting its teacher branch would leave that separate error. No such later fit is included in this first probe.

## Evaluation and interpretation

Evaluate the missing-term prediction and final waveform separately on all 96 development sources, including the targeted cases. Report sourcewise and teacher-defined regional results for quiet, near-silence, active audio, startup and expressive sources. The development panel is held out from this fit, but has been inspected previously; it is not a fresh final qualification set. Preserve the fixed quiet thresholds and their known limitations. Do not call micro-amplitude threshold crossings audible failures without listening evidence.

Compare the untouched initialization, oracle first restoration and folded affine correction with the same downstream initial weights. Report signed means and AC/RMS of the retained, discarded, predicted and residual terms. Negative cross terms can establish cancellation at that operation; final-waveform counterfactuals establish its downstream relevance. A component's large RMS alone establishes neither.

| Observation | Supported interpretation |
|---|---|
| A teacher ablation worsens waveform | That contribution matters in the teacher representation under this isolated intervention |
| Affine prediction generalizes and folded waveform improves | A zero-additional-operation initializer is promising at this location |
| Affine prediction is good but waveform worsens, as may the oracle | Local matching is insufficient for the downstream narrowed initialization; subsequent approximations still matter |
| Oracle helps but affine fit fails | The omitted term is not adequately recovered by this fixed linear predictor/data/ridge; nonlinear recovery or more capacity remains open |
| One operator's ablation has little effect | That isolated omission is less consequential on these sources; it does not certify every removed connection |

No result here proves an irreducible capacity limit, identifies the cause of updates 4,500→5,000, or establishes final codec quality. Teacher-only ablations, initial-student restorations and full-group recovery are different questions. Preserve all trained checkpoints, record source/checkpoint/code identities and verify that teacher and untouched initialization remain unchanged after every intervention.
