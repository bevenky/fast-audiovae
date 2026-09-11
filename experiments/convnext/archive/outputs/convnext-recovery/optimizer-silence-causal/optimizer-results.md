# What the optimizer comparison establishes

**Excessive displacement is a demonstrated contributor to quiet-audio regression in these four diagnostic updates. Optimizer history also changes the response. Peak overshoot remains unresolved.**

Each comparison started independently from checkpoint 8890. These were four separate single-step perturbations, not four consecutive training steps. Each batch shared the same discriminator update, loss-balancer state and clipped generator gradient between retained and fresh generator-state conditions. The ten held-out probes were fixed before comparison scoring.

Mean percentage change in error across the four updates follows. Negative means improvement. These are **MSE changes**, not RMS, correlation or listening-quality scores.

| Generator update | Natural quiet residual MSE | Encoded-zero quiet MSE | Full-scale peak-excess MSE |
|---|---:|---:|---:|
| Retained state, original displacement | +2.216% | +26.636% | +4.293% |
| Retained state, qualified smaller displacement | −0.199% | −6.724% | +0.260% |
| Fresh states, same qualified displacement | −0.751% | −13.179% | −0.088% |

The original displacement worsened both quiet metrics in all four batches. Rescaling the **same retained update direction** improved both in all four. This supports the earlier finding that an infinitesimally helpful direction can become harmful at its realized step size. Both normalized conditions also improved training waveform, mel and quiet errors in every batch.

Fresh states improved quiet more, but this clears **both Muon and AdamW generator histories**. It neither isolates AdamW nor establishes that an optimizer reset will improve continued training. Equal parameter-L2 displacement also does not mean equal acoustic change: the fresh condition moved training waveforms approximately **1.9–3.5 times more** than the retained condition.

Peaks tell a different story. Retained updates worsened peak-excess error in three of four batches even at the qualified smaller displacement. Fresh states worsened it in two of four. Their tiny mean improvement cannot establish a peak solution. Together with the preceding gradient diagnostics, this supports investigating update direction as well as magnitude for peaks; it does not prove all historical overshoots had one cause.

## Calibration and limits

Three separate training batches selected a common parameter-L2 radius of **0.002635961** using waveform-linearity, quadratic-error and observed-descent checks. This was 1/16 of the calibration reference displacement, **not an evaluated 16× learning-rate reduction**. No held-out outputs selected that radius.

Plain SGD failed the declared calibration grid, so the original three-condition comparison correctly stopped. A recorded amendment then compared only the two qualified native conditions on the four previously untouched batches. SGD was never scored on held-out probes; its failure does not prove smaller SGD steps cannot work.

All model, optimizer and RNG state was restored. No checkpoint was written or promoted. These results justify separating displacement control from history effects in the next controlled decision, not an automatic reset, learning-rate prescription or quality claim.

[Comparison evidence](two-native-comparison.json) · [Original calibration](optimizer-comparison.json)
