# No-update gradient audit

The four selected development sources were measured at steps 1000, 4500 and 5000. No optimizer step occurred. All 12 source MAEs reproduced saved validation, and target/cache, checkpoint, frozen decoder, teacher, gradient-slot, module-mode and RNG checks passed. Both diagnostic arithmetic tests passed locally and remotely. The Adam formula was checked against a real toy AdamW update.

## What the gradients establish

| Step-5000 observation | Measured result | Interpretation |
|---|---:|---|
| Four-source weighted waveform gradient norm | 0.385755 | Dominates this panel's ordinary parameter gradient |
| Weighted log-mel / feature / linear-mel norms | 0.001936 / 0.001385 / 0.000218 | These branches do not overwhelm waveform here |
| Waveform versus combined gradient cosine | 0.999983 | The combined raw direction almost matches waveform |
| Whistle waveform versus log-mel gradient cosine | -0.304054 | A real source-specific objective conflict |
| Spanish / Kannada near-wave versus feature gradient cosine | -0.163985 / -0.199435 | Matching all stage-4 features can oppose the near-silence subset |
| Actual late parameter delta versus active gain error | Positive on all three speech cases at both endpoints | The realized late change moved against local amplitude correction |

For all three speech cases at step 5000, both raw gradient descent and the hypothetical saved-state Adam update would reduce active gain error. This rejects a simple claim that mel or feature weights broadly force the late amplitude rise on these inputs. It does not explain which intervening fitting batches accumulated that rise.

The actual 4500-to-5000 parameter delta's step-5000 gain-error projection is dominated by convolution weights. For Kashmir, convolution contributes +0.0080271 of total +0.0082192; conditioning contributes +0.0001637 and Snake parameters +0.0000285. Its stage contributions are +0.0011461, +0.0024667 and +0.0046064 for stages 2, 3 and 4. Spanish and Kannada also have positive contributions from all three stages. These are local first-order projections, not an additive causal decomposition of 500 finite updates. They do not implicate the frozen suffix implementation or a single activation bug.

## Saved-momentum diagnostic

Negative derivatives mean local improvement; positive derivatives mean local worsening. Every per-source gradient below is its contribution using the same four-source denominators. The hypothetical update is not an actual training update and does not use a real fitting batch.

| Step-5000 diagnostic | Raw combined-gradient direction | Adam with saved moments | Same second moment, first moment reset to zero |
|---|---:|---:|---:|
| Spanish near waveform L1 | -7.34e-8 | +1.14e-8 | -5.49e-9 |
| Kannada near waveform L1 | -7.78e-8 | +9.83e-9 | -6.92e-9 |
| Whistle waveform L1 | -3.51e-5 | +9.50e-7 | -2.60e-6 |
| Whistle active log-gain error | -0.01006 | +0.001023 | -0.000660 |

Saved first-moment history reverses the immediate direction for these selected failures. This is evidence that optimizer history matters to local behavior, not evidence that a momentum reset will improve the full training run. At step 4500, the saved-state direction instead helps the two near-silence cases, so the effect is not a persistent one-way bias.

The pooled near-waveform gradient also has cross-source conflict: waveform contributes +1.41e-8 and stage-4 feature loss +7.09e-8 to its raw directional derivative, while log-mel contributes -1.70e-7, giving net -8.44e-8. Dropping spectral supervision indiscriminately would remove the strongest corrective direction for this near-silence panel. Near waveform reconstruction and the separate output-amplitude threshold are different objectives: the raw combined direction can improve the former while worsening the latter.

## The teacher has a small DC baseline, and the student drifts above it

All values below are signed waveform means in units of 1e-6, over unchanged teacher-RMS-at-most-1e-5 windows.

| Source | Teacher mean | Student 1000 | Student 4500 | Student 5000 | Window count |
|---|---:|---:|---:|---:|---:|
| Spanish | 9.533 | 6.138 | 16.295 | 14.864 | 44 |
| Kannada | 9.603 | 6.325 | 17.050 | 15.617 | 42 |

The teacher itself has a small positive near-silence baseline: its pooled DC component accounts for 98.1% and 99.7% of teacher energy in these Spanish and Kannada windows. The student also has AC residuals, so its entire near-silence error must not be labeled DC. The student changes from below it at 1000 to above it later. Both near-window residual means improve from 4500 to 5000, despite remaining above the teacher and failing the current amplitude checks. This differs from the broad active-audio gain regression over the same interval. Per-window means, RMS, limits and pass/fail values are preserved in the full JSON, including medians requested by the layer audit.

## Limits and next diagnostic

No implementation mismatch, detached student path, incorrect pooling denominator or frozen-state mutation was found by this probe. The evidence separates three problems: reduced-group approximation passed through an amplitude-sensitive frozen suffix, specific quiet/whistle gradient conflicts, and late fitting dynamics whose accumulated displacement is not explained by the current held-out direction. A focused audit of the actual last fitting batches and their saved optimizer directions would be needed before attributing the late drift to data order or choosing an optimizer change. No loss, learning-rate, architecture or optimizer change is recommended solely from this four-source counterfactual.

Raw evidence: `gradient-audit5000-v1.json`; compact calculations: `gradient-audit-summary.json`. The existing 96-source validation audit remains the population for this development panel. Four deliberately selected failures cannot establish language-wide or expressive-category performance.
