# What the projected hints establish, and the next reconstruction controls

The paired result is mixed. Compared with the matched baseline, projected hints improve complete-group MSE by 4.04%, but worsen waveform MAE by 0.174% and mel error by 0.511%. Quiet residual RMS improves by 0.292%, while failing quiet windows increase from 2,379 to 2,394 and failing near-silence windows increase from 181 to 183 out of 184. Neither arm has full-scale overshoots. This is not an overall quality improvement warranting adoption.

## Confirmed behavior

The feature objective did affect student learning. Both the trained readouts and the frozen initial readouts show lower development hint error. The improvement is therefore not solely the projectors adapting while the student remains unchanged. However, it establishes improved predictability through those readouts, not exact teacher-feature equivalence.

The feature/output disagreement is widespread: group MSE improves on 85 of 96 sources, but 48 of those 85 have worse waveform MAE and 42 have worse waveform MSE. The aggregate ratio of waveform MSE to group MSE rises from 0.0194611 to 0.0204025, a 4.837% increase. This is a descriptive ratio in two different coordinate systems, not a measured suffix Jacobian or amplification constant.

The frozen suffix is a nonlinear mapping. Reducing an unweighted group-feature error norm does not mathematically guarantee smaller waveform error. The saved averages do not identify which particular channels, phases or error directions caused that disagreement. We should not present that unmeasured attribution as established.

## The fixed hint strength varied across training batches

On the pooled 72-source calibration panel, the combined weighted hint gradient norm was 6.82% of the existing student-gradient norm, with cosine +0.06997. Each hint was calibrated to 5% on its own upstream student parameter set.

Fixed coefficients do not enforce those ratios on every batch. On the first actual update, weighted stage-3 and stage-4 hint norms were respectively 8.90% and 21.75% of the existing all-student gradient norm. On the last update, they were 13.21% and 60.38%. Individual norms cannot determine their combined norm without the cross term. Nevertheless, stage-4 guidance was substantially stronger on these sampled batches than the calibration percentage alone suggests.

Both sampled AdamW displacements have negative dot products with the existing combined objective and with each hint gradient. At the first update, where weights and moments match exactly, adding hints increases update norm by 2.078%, and the existing-objective gradient dot displacement becomes 1.116% more negative. Thus these records do not show a gross immediate ascent of the existing training objective. They also do not prove that the weighting was optimal or that no conflict appeared elsewhere. Only first and last component traces were saved, and the last-arm states differ.

## What remains unresolved

The comparison does not distinguish limited retained information from an imperfect initializer, an imperfect auxiliary metric, or downstream compensation for the narrowed features. All remain possible. The existing waveform and complete-group losses already supply teacher supervision; adding more teacher targets does not automatically solve these distinct problems. A 125-update comparison also cannot establish that projected distillation generally fails.

## Bounded native-upsampler experiment

Use two independent starting bases: the fresh sliced initialization and the preserved trained step-4625 student. Keep the source panels, whole-box interfaces, teacher and suffix fixed. The following controls answer different questions without neural retraining:

1. **Unchanged output and teacher-boundary oracles.** Compare the unchanged model with exact teacher stage-4-up output injected into its actual downstream residual stack. Separately inject the complete teacher stage-4-end output into the frozen suffix. The latter checks the external contract; the former tests whether the internal teacher coordinates still suit the adapted downstream units.
2. **Native operator reconstruction.** Fit the existing 128→128, kernel-4, stride-2 upsampler from actual student inputs to full teacher upsampler outputs on the 72 training sources. Retain exact causal/polyphase geometry, one shared bias and fixed ridge. Evaluate all 96 development sources. A teacher-selected-input fit is a local predictability comparison only, not a module to splice into the adapted student.
3. **Downstream compensation control.** For the trained-base fit, compare its output through the adapted stage-4 residual stack and the original teacher stack, which have the same 128-channel interface and operator shapes. The initial base already has the original stack. This distinguishes a locally improved fit from compatibility with downstream compensation; it does not itself propose replacing the trained stack.

An exact internal oracle can succeed where the fitted operator cannot because the oracle supplies unavailable information. Conversely, a teacher-coordinate fit can improve locally and damage a co-adapted downstream computation. The controls distinguish these outcomes. Failure of a fixed linear fit does not prove that the entire narrower nonlinear model lacks capacity.

Evidence: `results/hint-balance.json`, each arm's first/last records in `train.jsonl`, `hints-source1500.json`, and `development-source1500.json`. This review performed no model execution or training.
