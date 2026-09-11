# What the current evidence says about the compressed architecture

The changed stages 2–4 are the established location of the reconstruction error. We have not established which removed channels cause the remaining silence, whistle or gain errors, or that the narrower group has reached its capacity limit. The existing measurements support a sensitive approximation problem inside that group, combined with a demonstrated effect from the training update policy.

## What actually changed

The original three stages remain a single jointly trained group, with all nine residual units, their Snake activations, depthwise filters, channel mixing, conditioning and causal geometry intact. Internal widths changed from `512 → 256 → 128` to `256 → 128 → 128`. The group still receives the same 1,024-channel input and must reproduce the complete 128-channel teacher output. The downstream original decoder, neighboring-sample output convolution and tanh remain frozen.

The initializer selects teacher channel coordinates and slices the effective convolution weights; it does not preserve the original function automatically. At a mixing convolution, for retained coordinates K and removed coordinates D, the omitted term is `W_KD × features_D`. Later convolutions lose removed-input contributions too. This is an algebraic consequence of narrowing, not a newly discovered axis or padding bug.

Selection uses pivoted Cholesky on an uncentered activation Gram at the stage outputs. It measures representational coverage in that feature space, not downstream waveform sensitivity. It does not explicitly rank cancellation, output phase or quiet-audio influence, and it does not perform a downstream weight reconstruction. These limitations make a pruning-related explanation plausible; they do not establish which limitation is responsible for a measured failure.

## Evidence by failure type

| Question | What is measured | What remains unproven |
|---|---|---|
| Where does the error enter? | The shared prefix is exact. Full teacher stage-4 features through the frozen student suffix give bitwise teacher audio on all 18 inspected source/checkpoint combinations. | Which trainable unit or removed channel contributes the harmful direction. |
| Does the suffix magnify feature error? | Spanish active stage-4 RMS changes 0.35509→0.36240 from 4,500→5,000 while final RMS changes 0.038874→0.043613. Halving its group-output error cuts waveform residual RMS from 0.008508 to 0.004062. | A unique cause inside the group. Aggregate feature MSE can improve while the downstream result worsens. |
| Are quiet features zero? | No. On Spanish near-silence, teacher head-Snake RMS is 0.27537 while waveform RMS is approximately 9.62e-6. | Actual signed, weighted per-channel contributions to the small output. Large hidden features and small output alone do not prove removed-channel cancellation; filtering, small weights, bias and cancellation can all contribute. |
| Is near-silence an irreducible noise floor? | Residuals often improve while output-amplitude thresholds fail. At step 5,000, 43/44 Spanish and 41/42 Kannada failures are amplitude-only under existing checks. Kannada also has a separate larger startup transient. | Audibility of the tiny stationary deviations, or inability of the current widths to reproduce them. |
| Does narrowing permanently suppress whistles? | The same whistle's teacher RMS is 0.01835; student RMS is 0.02468 at 1,000, 0.01598 at 4,500 and 0.01633 at 5,000. Its amplitude bias changes sign. | A permanent amplitude ceiling or a particular removed phase representation. No dropped-channel phase-contribution measurement exists. |
| Is the late broad gain increase architectural? | Changing only accumulation improves full-panel waveform MAE 23.37% and substantially reduces speech gain excursions under matched source exposure. | That all remaining error is optimization-related, or that small-batch variance alone was the cause. The policy also changes Adam update count and moment history per audio exposure. |

## What the layer reports do not measure

The stage-2 and stage-3 diagnostics compare the student to **selected teacher coordinates only**. They do not compute the removed coordinates' outgoing contributions. The detailed residual-unit traces similarly compare activations and whole residual branches; they are not a kept-versus-removed contribution decomposition. A trained student may redistribute its representation, so even a large selected-coordinate error is not sufficient to label that layer defective.

The existing intervention changes the **entire full-width stage-4 error vector**. It establishes that the waveform error depends on the group output, but cannot assign it to pruning a particular channel, one residual unit, or phase information. There is no saved `W_KD × features_D` measurement, kept/removed cancellation cross-term, removed-channel ablation, or waveform change from restoring one such contribution.

If a causal channel attribution is required later, the missing measurement is explicit: at the original pruned initialization, use the same teacher inputs and causal history to separate retained and removed input contributions at the first affected mixing operation, then propagate a controlled restoration through the otherwise unchanged path. Measure signed DC, AC and final waveform changes separately. At a trained checkpoint, coordinate drift means a teacher contribution cannot simply be pasted in and called a fair repair; that would require a separate, properly defined intervention. No such experiment was run for this review.

## More training versus a capacity limit

There is no global plateau established by the saved trajectory. From 1,000 to 5,000, common mel error improves 32.32%, group MSE 53.19% and quiet residual RMS 27.58%; every active source improves correlation. The accumulation-12 comparison then reaches 97.6007% active correlation and better waveform error with the same widths. These results demonstrate remaining learnability and contradict a claim that the broad late gain problem required a new architectural layer.

They do not justify saying all failures will disappear with more training. Near-silence passes fell from 91/184 at step 1,000 to 4/184 at original step 5,000, and recover only to 18/184 in the larger-accumulation comparison. The single whistle's average gain error barely changes under larger accumulation. Laughter, breathing and other expressive recordings remain less accurate than ordinary speech. No converged matched-width experiment, reconstruction-aware initialization comparison, or wider-capacity control establishes whether these residual errors are an optimization, objective or capacity limitation.

One especially important caution: at step 1,000, the saved half-group-error intervention improves Spanish near residual RMS from 9.40e-6 to 4.22e-6, yet failed near windows increase from 2 to 43. Kannada residual falls from 17.47e-6 to 5.45e-6 while failures increase from 20 to 38. The residual and the tight absolute output-amplitude ceiling measure different things, and a nonlinear suffix need not move monotonically along a feature interpolation. Those failure counts alone cannot establish architectural degradation.

## Recommendation

Keep the causal group design and avoid attributing the remaining failures to a missing activation, phase layer or proven lost cancellation mechanism. The most defensible statement is that channel narrowing changed the learned function and its approximation errors reach sensitive downstream directions. The specific harmful removed contributions are still unmeasured. The improved accumulation policy earns continued investigation, but not a promise of eventual teacher equivalence or acceptance of the quiet failures. If an architectural intervention is considered, choose it only after distinguishing downstream-sensitive pruning from a training tradeoff, rather than adding layers on the basis of aggregate internal error.

Evidence: [pruning re-audit](pruning-reaudit.md), [layer diagnosis and raw report](layer-diagnosis.md), [raw layer measurements](regressing-layer-diagnosis-v1.json), [training history](validation-audit.md), [gradient audit](gradient.md), and [matched accumulation comparison](accumulation-comparison-v1/independent-analysis.md). Source mechanisms are in `group_model.py::_copy_conv/_narrow_stage`, `run_pilot.py::choose_channels/pivoted_subset`, and `boundary_diagnostics.py::evaluate_boundaries`.

This review reads existing code and saved reports only. It does not run models, training, benchmarks or tests.
