# Current distillation versus intermediate teacher hints

The current decoder already uses knowledge distillation. The specific missing mechanism is **projected intermediate feature guidance inside the compressed box**, not teacher supervision in general.

## What the current code does

- `run_pilot.py:176–196` computes raw teacher-waveform L1, multiresolution mel reconstruction, and raw MSE on the complete 128-channel stage-4 output. Feature cells are weighted by their valid waveform samples, including partial cells.
- `run_pilot.py:529–534` obtains the shared group input and target from the frozen teacher, executes the student's whole group, and backpropagates the combined losses. `group_model.py:305–308` keeps the frozen suffix differentiable, so waveform errors reach the trainable group.
- `group_model.py:205–208` exposes stage-2 and stage-3 teacher boundaries, but training only consumes `group_output`. The stage-2/3 comparisons in `boundary_diagnostics.py:1–5` are explicitly measurement-only.
- `settings_screen.py:300–301` records that the experiments retain waveform/full-boundary losses and exclude new stage-2/3 matching losses. The selected continuation uses that same objective, through `resume_settings.py:233–235,284`.

There is no evidence here that teacher targets were omitted or detached from the student-side loss graph. Intermediate diagnostic plots should not be described as intermediate distillation losses.

## What StreamCodec2 contributes

StreamCodec2 uses trainable linear projections to compare corresponding student and teacher intermediate features with different channel widths. It adds these hints to its reconstruction and adversarial training. Removing up/downsampling hints reduced ViSQOL from 4.313 to 4.296 in its experiment. Its distilled student still trailed the teacher, and the experiment used 16 kHz speech and a different codec. This supports testing intermediate guidance, not a guarantee of AudioVAE2 parity or a transferable loss coefficient. [StreamCodec2, §§2.3, 3.1, 4.2.2](https://arxiv.org/html/2509.13670v1).

## Smallest applicable addition

If testing hints after the reconstruction-aware initialization work, use at most two measured locations:

| Location | Student feature | Teacher feature | Training-only alignment |
|---|---|---|---|
| Immediately after stage-3 upsampling | 128 channels, 6 kHz | 256 channels, 6 kHz | Linear 128→256 |
| Immediately after stage-4 upsampling | 128 channels, 12 kHz | 128 channels, 12 kHz | Soft 128→128 projection, or a separately justified identity hint |

These are before their respective residual stacks. The existing loss on the full stage-4 **end** remains the external contract. Equal widths at the second hint do not prove that the student must preserve the teacher's internal basis. A projection permits representation adaptation, but it is not a deployed substitute for lost channels.

Targets must come from the pristine teacher on the same authenticated latent sequence. Student features must come from the actual student prefix, not teacher-forced internal inputs. Preserve exact rates and phases; use eight valid waveform samples per stage-3 cell and four per stage-4 cell, with the existing partial-cell weighting rather than interpolation or arbitrary trimming. Do not attach a separate target to every hidden layer.

Keep the projectors in a training-only sidecar and exclude them from export. They add training work but no decoder inference operations. The fixed encoder and shared latents do not need another encoder or codebook distillation objective.

## How to decide whether the addition helps

Keep waveform, mel and full-boundary losses active and unchanged in a matched control. Calibrate the added hint contribution on training data and check actual student-parameter gradients and optimizer displacement. Do not copy StreamCodec2's numerical coefficient across different reductions or normalize each quiet example by its tiny error. A jointly trained projector can absorb some mismatch itself, so a falling projected loss alone is insufficient evidence.

The decision remains teacher-relative waveform, gain, quiet DC/AC, transient and peak reconstruction on the reserved panel after a fixed recovery budget. Intermediate hints are a plausible way to localize supervision, but current results do not establish that their absence caused the step-5000 regression. No model, training or benchmark was run for this review.
