# Paired decoder diagnosis

Before this pass, the student stages and final teacher waveform had been checked, but a matched intermediate trace of both decoders was missing. That gap is now addressed for stationary silence and four selected natural recordings.

Both models received the same 64-channel latents. The trace covers the teacher's upsampling, conditioning, activation and residual stages, and the primitives inside all ten student ConvNeXt blocks. Instrumented outputs reproduce normal execution exactly; teacher scored outputs also exactly match the canonical training targets. No model or optimizer state was retained from any experiment.

| Problem | What the paired diagnosis establishes | Implication |
|---|---|---|
| Stationary silence pattern | The student's nearly constant final features become a repeating 480-sample waveform through its output-position weights. This component accounts for 93.19% of stationary residual power. The teacher's final learned multichannel convolution nearly cancels its periodic contributions. | A concrete synthesis mechanism is identified. It is not an unavoidable capacity limit for this fixture. |
| Loud peaks | Teacher pre-tanh maxima are 2.963 and 2.974 on the Sindhi and laughter cases, becoming 0.995 after tanh. Student maxima are 1.297 and 1.199 through its unbounded head. | The teacher's output bound is actively used. Adding tanh to a student already trained to predict compressed audio changes its learned mapping. |
| Last student block | Reducing block 10's contribution infinitesimally predicts lower peaks in both loud cases, but worse ordinary reconstruction there and worse quiet fidelity in all three cases with quiet samples. | The block carries useful information alongside a peak tradeoff. It is not a universally defective layer. |
| Quiet-error increases during updates | Full retained updates worsen quiet error on all four independent comparison batches; smaller moves in those same directions improve it. | Actual update magnitude is a demonstrated contributor. Optimizer-history effects require separate interpretation. |

Tanh leaves the teacher's stationary silence unchanged. Peak bounding and silence reconstruction are therefore separate issues. The result does not support a single activation or filter that solves both automatically.

The next focused design decision is how to preserve the useful last-block representation while training the output head to respect amplitude bounds, alongside controlled update magnitude for quiet fidelity. The existing checkpoint remains the reference. No new architecture, optimizer reset or training continuation was started.

The natural cases are deliberately selected examples, not a new multilingual quality benchmark. Block sensitivities are local derivatives at unchanged weights, not evidence that a finite gain edit will improve quality. Hidden feature values were observed within each architecture; they were not subtracted across incompatible feature spaces.

- [Stationary teacher/student results](paired-layer-results.md)
- [Every stationary layer observation](layer-inventory.md)
- [Natural quiet and peak results](natural-layer-results.md)
- [Bounded optimizer comparison](optimizer-results.md)
- [Architecture capacity and literature analysis](architecture-analysis.md)

Machine-readable reports: [paired layers](paired-layers.json), [natural cases](natural-layers.json), [optimizer comparison](two-native-comparison.json), [architecture checks](architecture.json).
