# Interpretation of the channel-contribution diagnostic

The results establish that removing cross-channel contributions changes the decoder substantially, even when all residual units, activations, strides and outer interfaces remain. They do not establish that cancellation is the universal mechanism, or uniquely explain the later behavior of the trained step-5000 student. These interventions use the original teacher and the fresh sliced initialization.

## What the interventions establish

On the 15-source intervention panel, restoring the teacher's discarded contributions at all eight affected mixing operations recovers the group output to normalized RMS error **4.45e-7** and the waveform to MAE **1.32e-8**, with **0/595** quiet-window failures. The full-width copied control is bitwise equal. This is strong evidence that the measured initialization gap is explained by the removed contributions and their propagation, rather than a different stride, final activation or suffix implementation.

This restoration requires teacher hidden features. It is an oracle diagnostic, not a deployable narrow decoder or evidence that the remaining channels can predict every discarded contribution.

The isolated teacher ablations show markedly different consequences:

| Removed contribution, otherwise full teacher | Waveform MAE | Active gain relative to teacher | Quiet failures |
|---|---:|---:|---:|
| Stage 2, first residual pointwise mixer | 0.004584 | 0.972 | 401/595 |
| Stage 3 upsampler | 0.014881 | 0.564 | 595/595 |
| Stage 4 upsampler | 0.028790 | 0.0689 | 595/595 |

These are controlled term removals inside the full teacher, not simulations of the entire pruned network. Nevertheless, they identify the later upsampler contributions as important to waveform amplitude and quiet reconstruction. The stage-4 intervention is the most consequential of the three tested.

## Cancellation is specific, not universal

For a selected linear output, the diagnostic splits it into retained-input contribution plus bias and discarded-input contribution. Their signed cross term distinguishes reinforcement from cancellation. A large discarded RMS alone cannot do this.

Pooled over sources, the total cross term is positive for the first stage-2 mixer and stage-3 upsampler in quiet, near-silence, active and startup regions. Those discarded contributions predominantly reinforce the selected output. At stage 4, all-quiet regions show modest cancellation: twice the cross term is **-0.002018**, reducing summed component energy by **6.24%**. But stage-4 near-silence shows reinforcement, **+0.024969**. In the Spanish and Kannada near-silence examples, a negative AC cross term is outweighed by positive DC interaction.

The whistle source demonstrates why these distinctions matter. At stage 4, retained and discarded terms cancel in its quiet/startup spans but reinforce in its active span. Source labels are not timed event annotations, and the startup metric covers only the absolute first 40 ms, not every interior onset.

The saved Spanish 10.24–10.28 s waveform excerpt directly shows different effects downstream. Removing the stage-3 contribution produces a residual mean about **2.06e-5** and residual AC RMS **1.60e-6**. Removing stage 4 produces residual mean **-7.48e-5** and AC RMS **7.66e-5**, with a strong repeating component. Thus both offset and periodic-waveform changes are observed. It would still be incorrect to infer that all original hidden channels cancel to silence or that this excerpt diagnoses the trained student's exact failure.

## Missing terms and upstream drift both matter

At the first stage-2 mixer, retained-input drift is only numerical noise, about **9.45e-8 RMS** pooled. This is the cleanest place to attribute the local difference directly to omitted inputs.

At later operations, the narrow prefix has already changed the inputs. At stage 3, the discarded term has RMS **0.1025** and retained-input drift **0.0635**. At stage 4 they are **0.1381** and **0.0624**. In near-silence the stage-4 values grow to **0.2475** and **0.0973**, and reinforce to produce a combined local gap of **0.3245 RMS**. Assigning that whole gap to the local missing channels would ignore upstream changes.

## The first affine reconstruction generalizes, but is not a whole-network repair

The fixed, training-only ridge fit uses all valid cells from 72 sources, weighted by valid waveform samples, with centered FP64 covariance and an unpenalized intercept. The retained-input covariance has full rank 256 and condition number about 2,753. The normal-equation residual is **8.47e-16**, so a failed numerical solve is not a plausible explanation.

On the 96-source development panel, the folded affine correction removes approximately **75.9%** of the missing term's uncentered energy overall and **93.3%** in near-silence. These are residual-energy reductions, not centered R-squared scores. Fit/development values are similar.

However, full-waveform MAE changes from **0.0248091** to **0.0248175**, and quiet RMS worsens from **0.0009704** to **0.0010267**. Even exact first-mixer restoration leaves the whole waveform strongly attenuated and all **2544/2544** quiet windows failing. Therefore, predicting that first term reasonably well is insufficient to repair the later missing paths. The existing development panel has been inspected before and is not a new final holdout.

## Smallest defensible next direction

Prioritize reconstruction of the later upsamplers, beginning with a bounded stage-4 probe, while keeping the same boundary contract and deployed operator shapes. Fit its existing stride-2, kernel-4 transposed convolution against the full teacher upsampler output using the **actual narrow-prefix input** and exact causal/polyphase alignment. Do not replace this with phase averaging or teacher-input fitting followed by student-input deployment.

If useful, reconstruct stage 3 and then refit stage 4, or optimize them jointly against fixed teacher boundary and waveform targets. Stage 4 must be refreshed when stage 3 changes its input. Judge the complete waveform and all regions, not just local feature error. Any correction should fold into existing weights and bias; no inference-time teacher or new adapter is implied.

A successful fixed-shape fit would show avoidable initialization error. Failure of one linear fit would not prove that the narrower nonlinear model cannot learn the task. These findings justify that controlled reconstruction question before wider layers, extra residual pruning or another unchanged long training run. No additional experiment was performed for this interpretation.

Evidence: `results/plain.json`, `affine-fit.json`, `fit-fitted.json`, `development-fitted.json`, `oracle_first.json`, `oracle_all8.json`, the three `teacher_ablate_*.json` files, `full-width-copy-control.json`, and `snippet-analysis.json`.
