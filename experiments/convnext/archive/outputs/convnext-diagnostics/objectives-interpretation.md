# Objective and optimizer diagnosis

The measurements narrow the problem to how a correction is expressed through the student and its finite optimizer update. They do not show that the losses stop rewarding teacher agreement, and they do not identify one universally harmful loss or optimizer.

This is one training batch of 32 crops, evaluated on 18 deliberately difficult held-out crops at three frozen checkpoints. It is a diagnosis, not a training-quality comparison. All disposable updates were restored; no retained checkpoint changed.

## Does the objective reward reaching the teacher?

Yes, along every tested student-to-teacher interpolation. With each checkpoint's current EMA coefficients fixed, the weighted objective decreases throughout the five tested interpolation points. For targeted, it falls from 59.490 at the student to 0.656 at 99% interpolation and 0.00244 at the exact teacher waveform. Quiet, active, original-overshoot and teacher-high-peak regions all have a favorable selected gradient projection toward the teacher before exact equality. This is an output-space test; the interpolation is not proof that the student can realize those outputs with a parameter change.

At the teacher, waveform, mel and feature-matching losses and their returned gradients are exactly zero. The adversarial loss and its audio gradient remain nonzero. That is a real residual adversarial force, but it does **not** prove that the teacher cannot be a local optimum: waveform and feature-matching losses have L1 cusps, and automatic differentiation selects zero subgradients at exact equality. Their true one-sided cusp cost must not be omitted when interpreting the teacher-point gradient. Nonzero GAN loss alone is also not evidence of conflict.

## What does one native update actually do?

The table shows the observed change after the ordinary D-then-G step on a disposable copy. These are equal-crop mean changes on the selected panel. Quiet error means squared error, not RMS; peak-excess energy is `mean(relu(abs(student) - max(1, abs(teacher)))²)`.

| Checkpoint | Quiet squared error | Overall waveform squared error | Peak-excess energy |
| --- | ---: | ---: | ---: |
| Parent 8,090 | −16.49% | +0.259% | +31.46% |
| Targeted 8,490 | +34.96% | +0.741% | +17.54% |
| Complex 8,490 | +49.02% | +0.578% | +12.48% |

For the fixed-D all-loss replay, quiet error improves in all 12 crops containing quiet windows at the parent. It worsens in 11 of 12 at targeted and 10 of 12 at complex. Some crops contain only one 20 ms quiet window, so these equal-crop averages are not duration-weighted estimates of corpus quality.

The first-order parameter-gradient prediction says quiet error should decrease at all three checkpoints. The finite update instead increases it at targeted and complex. The fixed-delta interpolation below confirms that a locally helpful direction exists and that its quiet benefit reverses at the full displacement. This is stronger evidence than the previous positive audio-gradient cosine, but does not identify the best learning rate or isolate the source of the higher-order response.

Updating D immediately before G changes these results very little compared with holding D fixed. That particular ordering does not explain the observed degradation on this batch.

## Which terms matter in the replay?

- Removing the current adversarial or feature-matching gradient barely changes the result. Neither omission fixes the quiet regression. Their earlier influence remains in the learned weights and optimizer moments, so this does not establish that GAN/FM training is unnecessary.
- Removing waveform reconstruction makes overall held-out squared error substantially worse. It reduces peak excess in targeted but sacrifices reconstruction. The waveform term is providing useful corrective information, despite a remaining peak tradeoff.
- Removing mel slightly reduces the overall squared-error increase, but worsens quiet error and peak excess in targeted and complex. No single loss removal is a clean solution.
- Even with all current gradients set to zero, the saved optimizer history and weight decay produce a substantial update. Targeted quiet error increases 42.0% and overall squared error 2.99%, versus 35.0% and 0.741% with all current gradients. This establishes that the optimizer state materially affects the next update; it does not isolate momentum from decay or prove that an optimizer reset would help.

Leave-one-out effects are nonadditive because clipping and adaptive optimizers operate on the combined gradient. They are not percentages of responsibility.

## Where do the gradients go?

On the held-out routing panel, the largest combined parameter-gradient norm is in block 9, followed by block 8, at every checkpoint. At targeted, block 9's norm is 538.4 versus 27.1 in the output projection. Quiet-origin gradients are concentrated late in the network. This does not support treating the final projection as the only part involved.

Quiet-origin and active-origin parameter gradients have positive aggregate cosine, about 0.18 to 0.24. There are local conflicts, including a targeted adapter cosine of −0.227, but no broad sign of global quiet-versus-speech cancellation. The plain-gradient first-order direction improves quiet error; the actual finite adaptive update remains the critical distinction.

AdamW-routed parameters have the larger raw gradient norm on the routing panel, while Muon-routed matrices have the larger aggregate parameter-change norm in the native replay. These are different batches and neither norm is an audio-impact attribution. Layerwise response and displacement diagnostics must determine which changes cause the undesirable audio behavior.

## Does a smaller displacement follow the predicted direction?

Yes. We executed the same native D-then-G step once per checkpoint, cached its parameter delta, and evaluated fractions of that fixed delta. The optimizer was not rerun at different rates. Student buffers stayed unchanged, every zero-displacement endpoint reproduced the baseline, and full-displacement metrics matched the previous exact-step replay.

| Fraction of the same parameter delta | Parent quiet MSE | Targeted quiet MSE | Complex quiet MSE |
| --- | ---: | ---: | ---: |
| 0.10 | −3.93% | −3.24% | −2.72% |
| 0.25 | −8.87% | −5.23% | −3.52% |
| 0.50 | −14.58% | −1.01% | +3.68% |
| 1.00 | −16.49% | +34.96% | +49.02% |

The quiet regression at the two later checkpoints is therefore a finite-displacement effect on this panel, not evidence that there is no useful descent direction. The error metric's own quadratic terms and the student's nonlinear response both contribute. This is one fixed update direction, not a validation of a smaller training learning rate over time.

Peak excess increases even at the 0.10 fraction: +2.82% at parent, +1.62% at targeted and +1.14% at complex. Scaling this one update down reduces the damage but does not reverse its peak direction. Silence and peak overshoot need separate explanations.

## Which parts of this same update change the audio?

Applying cached parameter subsets gives direct counterfactual responses, beyond gradient norms. Each subset starts from the exact baseline. The first three rows partition all parameters; the final two form a separate optimizer-routing partition.

| Applied subset of the cached full delta | Targeted quiet MSE | Complex quiet MSE |
| --- | ---: | ---: |
| Output projection only | −0.53% | −0.22% |
| Blocks 8 and 9 only | +14.44% | +27.96% |
| All remaining parameters | −4.71% | −5.08% |
| Muon-routed parameters only | −0.15% | +1.91% |
| AdamW-routed parameters only | +0.44% | +8.35% |
| Entire delta | +34.96% | +49.02% |

Late-block movement can harm quiet audio while improving overall waveform MSE: blocks 8 and 9 alone improve overall MSE by 0.176% at targeted and 0.271% at complex. This implicates their response and its interaction with other parameter changes, not the output projection alone. At the parent, those same late-block updates improve quiet MSE by 19.32%, so the layers are not intrinsically defective.

The combined Muon-plus-AdamW quiet change is far worse than either subset alone. That establishes a substantial interaction in their joint finite displacement, including nonlinear model effects and cross terms in squared error. It does not identify either optimizer as universally wrong. For peaks, the Muon-routed subset carries a larger adverse effect in these later checkpoints, but routing also selects a particular parameter family; this is not a comparison against AdamW updating those same matrices.

The current evidence supports investigating finite update size, optimizer history, and late-layer response before changing the architecture or adding more losses. It does not support blaming clipping, Muon, AdamW or GAN universally, or choosing a new learning rate from this one batch.

Evidence: [objective and native update measurements](objectives-updates.json), [fixed update path and parameter-subset effects](native-step-path.json), [recorded optimizer history](optimizer-history.json).
