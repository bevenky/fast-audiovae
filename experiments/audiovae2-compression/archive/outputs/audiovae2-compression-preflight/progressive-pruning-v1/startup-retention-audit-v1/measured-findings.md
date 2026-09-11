# Startup retention: measured four-update diagnosis

11 September 2026. Analysis of saved aggregate evidence only. No model execution, training, source modification or threshold change was performed for this note. The recommendation is **one startup-only persistent-anchor experiment from fresh C**, preserving the completed G candidate and the original current-batch Q protocol as separate evidence.

## What is now established

The [probe](probe-v1-aggregate.json) completed four ordinary AdamW updates from authenticated fresh C, using 48 distinct sources in the original order. Development startup passing fell from 13/13 to 2/13. All 11 new failures were **amplitude-only**: every startup window still passed its waveform-residual limit. Calibration passing fell from 6/6 to 2/6, also entirely amplitude-only.

The original waveform, mel and group-feature objective, FP32 policy, singleton execution, accumulation of 12 sources, and AdamW settings were retained. All 96 preparation/training cache comparisons and all 19 startup-target comparisons were exact. The probe restored fresh C weights, empty optimizer state and original RNG; teacher, frozen student parts and protected files were preserved. No checkpoint was retained. This establishes update-induced loss of the measured startup property on this short trajectory. It does not establish a general capacity limit, a unique loss-branch cause, or what freezing a block would do to ordinary recovery.

Continuous values below are pooled directly from FP64 sample energies. Passing uses the unchanged evaluator's FP32 RMS and its original bounds. `microFS` means one millionth of waveform full scale; these values are not perceptual loudness measurements.

| Update | Calibration passing | Development passing | Development residual RMS, microFS | Development output RMS, microFS | Residual DC RMS, microFS | Residual AC RMS, microFS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6/6 | 13/13 | 1.955965 | 9.405918 | 0.369101 | 1.920824 |
| 1 | 6/6 | 13/13 | 2.091138 | 9.389918 | 0.817682 | 1.924644 |
| 2 | 6/6 | 13/13 | 2.166334 | 9.584830 | 0.880855 | 1.979165 |
| 3 | 6/6 | 13/13 | 2.305175 | 10.184243 | 0.991397 | 2.081096 |
| 4 | 2/6 | 2/13 | 2.576578 | 10.641997 | 1.297277 | 2.226169 |

The development teacher RMS remains 9.333619 microFS. Residual signed mean changes from −0.068586 to +1.189428 microFS. Both DC and AC error increase; reducing this to a pure bias/DC fault would be incorrect. Two of the four ordinary batches contain no near-silent startup window; the others contain one each. Presence in some batches plainly does not make the global objective a preservation constraint.

## Cold numerical behavior is a separate finding

The [numerical audit](numerical-v2-aggregate.json) examined 12 fitting sources, eight with quiet windows. One source's cold gradient-enabled path differed from its comparison path: maximum waveform difference 1.527369e-7, 104 constraint scalar differences with maximum 8.867304e-14, and **zero acceptance changes**. The first different recorded module was `model.3.block.0`, the stage-2 input Snake. After warmup, the recorded repeated gradient/no-gradient layers, waveforms and constraint scalars agreed exactly; two metric paths on the same prediction also agreed.

This localizes the observed cold execution difference; it does not identify an upstream PyTorch defect or prove all shapes harmless. The four-update probe qualified its execution paths before measuring drift, and startup then failed under real parameter changes. Thus the cold mismatch explains Q's strict pre-state rejection, while it does not explain away the measured retention failure. The warmup is an explicit execution qualification, not a claim that this diagnostic reproduces every historical cold runtime decision.

## Eight parameter interventions localize the early interaction

The probe mixed initial and step-4 **parameter states** within the same student. It did not inject teacher hidden channels into an adapted basis. The disjoint partition covers all 90 trainable tensors:

- **P, 33 tensors:** stage 2 and the stage-3 input conditioning/Snake.
- **U, 3 tensors:** the native stage-3 upsampler.
- **R, 54 tensors:** stage-3 residual units, stage 4 and its conditioning. R is trainable downstream group content; the outer frozen suffix is unchanged throughout.

| Partitions taken from step 4 | Calibration passing | Development passing | Development residual RMS, microFS | Development output RMS, microFS |
|---|---:|---:|---:|---:|
| None | 6/6 | 13/13 | 1.955965 | 9.405918 |
| R | 5/6 | 12/13 | 2.279648 | 10.393522 |
| U | 6/6 | 13/13 | 2.218592 | 9.391101 |
| U + R | 6/6 | 12/13 | 2.423644 | 10.361503 |
| P | 6/6 | 13/13 | 2.005365 | 9.675823 |
| P + R | 2/6 | 2/13 | 2.410790 | 10.663342 |
| P + U | 6/6 | 13/13 | 2.290942 | 9.671223 |
| P + U + R | 2/6 | 2/13 | 2.576578 | 10.641997 |

Restoring original R while retaining the updated P and U restores all 13 passes. Restoring only U while retaining updated P and R leaves 11 failures. U movement is therefore unnecessary for these failures, and protecting only the fitted upsampler would miss the observed interaction. This is a conditional intervention on this four-update trajectory, not an additive attribution or proof that R should remain frozen during useful recovery. The hybrid ordinary-audio quality was not measured.

The U+R hybrid also provides an unusually direct limitation: **all six calibration anchors can pass while a development startup fails**. Finite-anchor preservation cannot be advertised as a guarantee for the 13 development windows, much less deployment audio.

## Why a first-order constraint prediction can miss this

Initial calibration residual RMS is only 1.059556e-9 FS. Its selected maximum residual-excess gradient norm is 5.016620e-13. The first actual Adam displacement predicts a change of −2.522445e-17 in that selected residual constraint, but the same window actually increases by +1.150533e-13. The new maximum increases by +4.061911e-12 and occurs at a different window. Near a nearly exact fit, a squared-residual gradient can be tiny despite sensitivity to a finite displacement. This is consistent with higher-order effects; the recorded scalars do not uniquely separate curvature from all finite-precision effects.

For amplitude, the initially selected window's prediction is −2.121340e-11 and its actual change is −1.673167e-11, so that local prediction has the correct direction. A different window becomes the maximum, increasing the maximum by +1.931182e-13. Both first-update selected gradient dot products are negative. **The probe does not prove positive first-order gradient conflict.** It demonstrates that one selected gradient does not reliably describe the finite-step maximum over several windows. Loss-component gradients were not separately measured, and these directional measurements concern update 1, not update 4.

Nor should ordinary updates be assumed to improve speech on every step. On the same first fitting batch, total ordinary objective rises from 0.01053158065 to 0.01434311846, **+36.1915%**; waveform, mel and feature losses all rise. This is one finite proposal at one state, not evidence that Adam is generally unsuitable. It means anchor preservation and useful ordinary learning must both be demonstrated, rather than treating all preserved Adam movement as beneficial.

## Recommended next experiment and decision limits

Use fresh C, its original teacher-derived initialization, fresh AdamW, the same ordinary losses and the same ordered 24,000 distinct fitting sources. Add only persistent startup protection during the approved bounded 2,000-update recovery. Keep all 90 group parameters trainable. This directly tests retention; starting from G instead would change both initialization and update policy relative to C.

Use the existing six calibration startup windows, never the 13 development windows, to define two separate maxima: residual excess and output-amplitude excess. Both caps are zero because every anchor initially passes. Refresh the relevant gradients at current parameters and act on the **actual Adam displacement**. Accept a proposal only after the complete six-window set passes the unchanged nonlinear evaluator. With two maxima the linear subproblem needs at most two rows. All-window nonlinear rescoring, not a gradient sign or a quietly enlarged tolerance, is what establishes finite-anchor feasibility.

One gradient at a changing or tied maximum can still produce unnecessary backtracking or zero movement. A tiny residual gradient need not produce a useful corrective direction. Record accepted displacement norms, fractions, zero-movement counts and correction norms, alongside ordinary quality and all seven quiet cohorts. If Adam moments advance when the parameter displacement is zero, record that policy explicitly and distinguish optimizer steps from nonzero parameter updates. Do not claim a successful recovery merely because anchors remain passing while learning stalls.

Leave the six current-batch Q constraints out of this first retention arm. They add ordinary-quiet, other-near and current-startup objectives whose incremental value the failed original Q never measured. Preserve Q and its failure evidence; a later all-quiet extension is justified only if the simpler retention arm yields useful learning and a concrete remaining quiet problem. Retain G as an independent promising candidate, without adding anchors to its adapted endpoint or promoting it now.

The [completed G comparison](../grail-hidden-v1/completed-aggregate.json) is matched to B at 2,000 updates: waveform MAE 0.00247640320 versus 0.00261464824, about 5.29% better; 80/96 recordings improve. Its group MSE is about 1.59% worse. G passes 1,419/2,544 quiet windows versus B's 1,242, but still passes 0/13 startup windows, and its 20–40ms teacher-transient cohort is worse. G supports a better recovery tradeoff from folded initialization, not successful startup retention or universal superiority. The development panel is reused, not an untouched final test.

Recurring calibration anchors are training involvement, even when absent from the ordinary loss. The user's existing data-reuse authorization covers a declared anchor trial. Keep the ordinary 24,000-source ledger and repeated six-anchor exposure/compute ledger separate; do not call all uses unique or mere monitoring. This adds training work while retaining the deployed architecture and native operators. It makes no new CPU latency or perceptual-quality claim.

The relevant primary-method rationale and its limits remain in [literature.md](literature.md). The completed probe resolves its proposed numerical and short-trajectory questions; it does not yet establish that the recommended constrained update will preserve useful learning or generalize beyond the six anchors.
