# Teacher compression: mathematical and design audit

This read-only review inspected the pinned teacher, the actual compressed-group constructor, channel selection, loss code and existing diagnostic reports. No model, training, test or benchmark ran. **I found an important approximation in initialization, but no new demonstrated indexing, conditioning, latent-context or weight-normalization defect.** Passing implementation checks does not establish that the chosen compression will recover the teacher's function.

## What is and is not preserved

The original group is stages 2 through 4: `1024→512→256→128`, at 200, 1,200, 6,000 and 12,000 Hz. The student is `1024→256→128→128`. Both retain all nine residual units, all dilations, causal padding and transposed-convolution phase conventions. The whole group's input and complete 128-channel output share the teacher's coordinates. The internal stage-2 and stage-3 outputs do not share the teacher's full interface and are not required to match selected teacher coordinates during training. Source: [group construction](../../../work/fast-audiovae/experiments/audiovae2-compression/group_model.py#L351) and [full-boundary objective](../../../work/fast-audiovae/experiments/audiovae2-compression/run_pilot.py#L176).

| Potential fault | Current evidence and remaining limit |
|---|---|
| Wrong normalized weights | Effective weights are recomputed from the live legacy hook before slicing. New scales are reconstructed from the sliced matrix, with zero-row handling. The code does not retain incompatible old normalization scales. This is a correct effective-weight migration, not preservation of the original optimizer geometry. |
| Wrong channel axes | Transpose weights use input/output axes 0/1; ordinary pointwise weights use output/input axes 0/1; depthwise, Snake and conditioning select the corresponding coordinates. The lost contributions discussed below remain absent even with correct axes. |
| Conditioning applied twice or to wrong coordinates | The group starts before stage-2 conditioning. Teacher and student each apply it once. Stages 3 and 4 inherit conditioning on their selected input coordinates. Both use the 48 kHz bucket. Quality of other conditioning buckets is not established by this 48 kHz training. |
| Latents or history changed | The frozen encoder's full-source raw 64-channel latents are preserved. Interior crops retain 30 frames; the decoder requires at most 20 previous latent frames. All 1,500 replayed sources and both accumulation arms reproduce cached teacher targets bitwise on scored samples. |
| Detached or broken suffix | The suffix parameters are frozen, but its student input remains differentiable. Feeding the real teacher group output through it reproduced the teacher waveform bitwise on all 18 previously measured source/checkpoint combinations. |
| Missing teacher output protection | Original final Snake, neighboring-sample convolution and tanh are retained. Both accumulation endpoints had zero full-scale overshoot samples. Tanh does not enforce silence accuracy, gain calibration or teacher peak shape. |

Code evidence: [effective WN migration](../../../work/fast-audiovae/experiments/audiovae2-compression/group_model.py#L37), [copy axes and residual units](../../../work/fast-audiovae/experiments/audiovae2-compression/group_model.py#L228), [conditioning and group execution](../../../work/fast-audiovae/experiments/audiovae2-compression/group_model.py#L169), [frozen-suffix differentiation](../../../work/fast-audiovae/experiments/audiovae2-compression/group_model.py#L305). Numerical evidence: [layer diagnosis](layer-diagnosis.md), [context audit](fresh-target-consistency-source-audit.md), and [completed accumulation comparison](accumulation-comparison-v1/execution-audit.md).

## The important initialization gap

For one teacher residual unit, write its channelwise Snake/depthwise/Snake path as `φ`. Partition channels into retained `I` and removed `J`. Its selected output is

`y_I = x_I + W_II φ(x_I) + W_IJ φ(x_J) + b_I`.

At corresponding input coordinates, the copied smaller unit initializes as

`s_I = x_I + W_II φ(x_I) + b_I`.

**The removed cross-channel term is not reconstructed.** The same problem occurs at the next upsampler: its omitted input columns formerly contributed after conditioning and Snake. Correct weight slicing only preserves the terms that remain. Even if a removed activation were predictable from retained activations, the implementation does not fold that prediction into the retained outgoing weights or bias.

Channel selection uses uncentered teacher stage-output Gram matrices and deterministic coordinate pivoting, sampling at most 256 time cells per calibration source. It does not use outgoing weights, suffix sensitivity, transient/event windows or measured teacher-function reconstruction to choose channels. It returns indices and a method string, not singular spectra, retained error or a recovery certificate. The same stage-output selection is then used inside each of that stage's three residual units, whose intermediate features need not have the same redundancy. These are limitations of the chosen initializer, not evidence the chosen coordinates are necessarily poor. Source: [selection implementation](../../../work/fast-audiovae/experiments/audiovae2-compression/run_pilot.py#L207).

Uncentered energy can emphasize large means or high-energy coordinates. It does not specifically protect low-energy components that matter after the frozen decoder's cancellation or mixing. Random temporal cells can also underrepresent short events. A source-level expressive label is not proof the sampled cells contain the expressive event.

This gives a reason to test **reconstruction-aware initialization at the same widths** before paying for a wider graph. Fit existing outgoing affine weights and biases from actual compressed upstream features to the teacher target on fitting-only data, accounting for already accumulated approximation. Joint whole-group training would follow with the same external target. This would add no deployed operation, but its benefit is unmeasured. It must be a separate fresh teacher-derived comparison, not an overwrite of the trained candidate.

## Do not call the channel reduction a proven information bottleneck

Per-frame width alone is insufficient: the student maps 1,024 × 200 Hz = 204,800 feature values per second to 256 × 1,200 Hz = 307,200, then to 128 × 6,000 Hz = 768,000. Its upsampling phases and temporal state can preserve information despite smaller channel counts than the teacher. The actual data also originate from a much smaller 64-channel latent sequence, not arbitrary independent values at every hidden coordinate.

These counts do not prove injectivity or adequate capacity. They show why “half the channels means half the information is gone” is not a valid rank argument here. We have not measured the relevant input-manifold rank, phase-aware approximation error or best achievable same-width teacher-function fit. What is demonstrated is the removal of terms from the copied initialization, followed by imperfect recovery.

## Why quiet and whistle errors remain plausible without a software bug

The measured Spanish near-silence teacher head features have RMS 0.27537 while waveform RMS is about 9.62e-6. Small feature-combination errors can disturb a cancellation and create a micro-amplitude offset. Aggregate stage-4 MSE can improve while the waveform mean shifts. The existing group loss treats all full-boundary coordinates uniformly; it is not a bound on error after the nonlinear suffix. End-to-end waveform loss supplies the required downstream gradient, but the prior gradient audit found source-specific conflict with feature or spectral objectives. This is evidence of sensitivity and tradeoffs, not proof that a particular channel or layer must be added.

Larger accumulation substantially improved speech waveform error and reduced gain excursions with the **same architecture**. That argues against explaining the entire problem as an irreversible capacity limit. However, quiet RMS improved only 0.12% relative to control, 2,406 of 2,544 quiet windows still fail, and the whistle's trajectory-average gain remains around 0.87 of the teacher. Its smaller excursions do not fix its persistent bias. Rare expressive exposure and source-specific gradient conflict remain competing explanations. [Independent comparison](accumulation-comparison-v1/independent-analysis.md).

## When to change course

The present evidence supports a bounded continuation with the better accumulation policy, not an assurance that enough steps will cure every failure. Predeclare the next fixed-exposure review and inspect quiet residual/DC, whistle level and error, source-level spectral fidelity and complete boundary error as well as the mean waveform score. If those failures remain effectively unchanged while only global metrics improve, stop using more training time as the sole proposed remedy.

Before widening, distinguish three cases:

1. **Same-width reconstruction improves the initial and recovered teacher function:** the method of compression was leaving recoverable contributions behind. Keep the cheaper graph if held-out audio improves too.
2. **Representative fitting examples recover but held-out failures do not:** investigate coverage and generalization. A wider graph is not the established remedy.
3. **Representative fitting and held-out group/waveform errors remain after stable recovery, while one controlled width restoration improves both:** that is useful engineering evidence for insufficient capacity at that boundary. Select one minimally wider boundary using downstream reconstruction evidence, keeping the rest fixed.

A better initializer cannot explain the sudden late amplitude fluctuation retrospectively; the accumulation comparison already demonstrates a training-policy contribution. Conversely, that policy's success does not certify the current channel selection or make persistent quiet/whistle errors acceptable. Residual-unit deletion would remove more nonlinear capacity and history and is not supported as the next quality fix. Existing conditional alternatives and their static, unmeasured CPU budgets remain in [the options memo](conditional-student-options.md).
