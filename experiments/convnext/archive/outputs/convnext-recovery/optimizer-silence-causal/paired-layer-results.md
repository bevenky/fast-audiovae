# What the paired layer trace shows

We now have a verified trace of both frozen decoders on the same six-second encoded-silence latent sequence: 120 teacher observations and 80 student observations, including the individual operations inside residual blocks. Before this investigation, the student stages had been traced, but the teacher's intermediate stages had not been traced alongside them.

The result identifies a concrete mechanism for the stationary silence pattern. It does not establish that one computational layer is malfunctioning or that this mechanism explains every quiet recording and overshoot.

## Exact execution checks

The same raw encoder means, shaped `1 × 64 × 150`, entered both models. No new encoding, resampling or latent scaling was introduced. Both instrumented paths reproduce their ordinary outputs exactly, and the ordinary teacher reproduces the frozen training target exactly after execution warmup. Model states and the step 8,890 checkpoint remained unchanged.

The first teacher execution differed from the cached target by a maximum of 6.43e-9 and RMS 1.30e-9. Its second, third and fourth ordinary executions were bitwise identical to the target and to the manual trace. This establishes execution-startup dependence. It does not isolate TorchScript, cuDNN or another backend as the cause. The discrepancy was recorded, not hidden with a relaxed tolerance. It is far below the student's stationary residual RMS of 4.38e-5.

## What happens between layers

The stationary comparison uses seconds 2–6, after both startup and the student's receptive-field history have settled. Every recorded stage repeats exactly across complete 40 ms cycles.

| Location | Teacher | Student |
|---|---|---|
| Decoder input | Same stationary, nonzero 64-channel latent at 25 Hz | Same latent |
| Early processing | Initial convolutions retain a constant temporal response | Adapter turns each latent into four 100 Hz phase vectors |
| First synthesis expansion | First transposed convolution introduces temporal phase structure | Four adapted phase vectors introduce 40 ms structure |
| Main network | Six upsampling stages reach 48 kHz through residual blocks | Ten ConvNeXt blocks remain at 100 Hz; later blocks strongly suppress phase differences |
| Final learned synthesis | 32 waveform-rate feature channels feed a learned seven-tap convolution | A 2,048-channel frame feeds 480 separate output-position weights |
| Final activation | Tanh changes stationary silence by exactly zero in FP32 | No bounded final activation |

The teacher also develops periodic structure internally. It has learned to suppress it by the output. The student is therefore not uniquely problematic simply because its hidden activations have phases.

Hidden channel amplitudes cannot be subtracted between the architectures to find reconstruction error. Their feature bases, widths, scales and time resolutions differ. Even a changing phase-power fraction within a model can reflect a change in its constant component, rather than amplification of the periodic component.

## The output heads give a concrete explanation

The student computes:

`waveform[480*n + r] = dot(output_weight[r], hidden[n])`

Even when `hidden[n]` is constant over time, the 480 different weight rows can produce 480 different sample values. Repeating that frame creates a 10 ms waveform pattern. The measured student-minus-teacher stationary residual consists of **93.19% shared 480-sample periodic power**, **5.48% DC** and **1.34% additional 40 ms structure**. The common feature vector, without its four-phase differences, accounts for the first two components. The remaining four-phase differences at the final head do not account for the dominant component. Upstream adapter effects on the phase-mean features are not ruled out.

For the teacher, a constant vector of 32 waveform-rate channels produces a constant output through its final convolution, away from boundaries. The actual learned seven temporal-tap contributions to the nonconstant component nearly cancel: their separate powers sum to 9.96e-10, while their coherent sum has power 4.14e-14. This is an algebraic description of the existing learned projection, not a measured 99.996% quality improvement from adding a filter. It is also not equivalent to the single-channel postfilter tried previously.

The teacher's final stationary waveform has RMS 9.63e-6 and is 99.955% DC power. Tanh leaves it unchanged at these amplitudes. The teacher's quiet behavior here comes from the learned synthesis path, not output clipping.

The head mechanism is consistent with research showing that convolution followed by a periodic channel shuffle can create tones because output positions use different filters. It does not require uneven transposed-convolution overlap. [Pons et al., upsampling artifacts](https://arxiv.org/abs/2010.14356), [authors' subpixel-convolution explanation](https://github.com/DolbyLaboratories/neural-upsampling-artifacts-audio/blob/main/ARTICLE.md#4-subpixel-convolutions)

## What this does and does not justify

The earlier rank test established that an alternative readout could represent this teacher silence target using the student's frozen final features. This rules out an unavoidable capacity barrier for this fixture. It does not show that fitting silence alone preserves other audio.

The evidence points to how the current features and readout have been learned together. It supports investigating their training and output sensitivities before adding another layer. Neither wholesale replacement with the teacher's expensive decoder nor a generic final filter follows from these results.

Natural quiet audio and loud transients require their own traces. Their changing waveforms must be evaluated at actual sample times, not reconstructed from stationary phase averages.

[Every recorded layer](layer-inventory.md) · [Full paired evidence](paired-layers.json) · [Architecture feasibility checks](architecture-analysis.md) · [Optimizer comparison](optimizer-results.md)
