# Matched-exposure accumulation comparison

The user approved this diagnostic after the late-update replay. Main training remains paused at 5,000. The experiment tests a practical batching policy without changing the model, loss definitions, learning rate or starting optimizer state.

## Locked comparison

| Item | Both arms |
|---|---|
| Initial model and AdamW state | Original step 4,500 checkpoint |
| Training examples | The same next 1,500 distinct cached sources, in the same order |
| Forward execution | One source at a time, with the existing causal context and valid-sample masks |
| Loss | Original waveform, mel and complete stage-2-to-4 boundary objectives and coefficients |
| Learning rate | 0.00003, unchanged |
| Precision | Existing FP32 path; TF32 disabled |
| Architecture | Existing narrowed stage-2-to-4 group; encoder, teacher and suffix frozen |
| Main run / checkpoints | Preserved; isolated output directories |

The control pools three sources per optimizer update, making 500 updates. The other arm pools twelve, making 125 updates. Both consume 1,500 sources. Their final optimizer counters are therefore 5,000 and 4,625 respectively. A common graph axis must be additional sources consumed, not optimizer step.

Loss contributions use the existing sample/element denominators across each accumulated group. A larger accumulation must not accidentally multiply the effective loss by four. Existing optimizer momentum is restored for both arms. At equal source exposure, accumulation also changes optimizer-update count and Adam's history measured in examples. This comparison cannot attribute a difference solely to gradient variance.

## Measurements and interpretation

- Evaluate the same four diagnostic recordings every 60 training sources, including zero and 1,500, for 26 aligned observations per arm. These recordings do not enter optimization.
- Evaluate the full 96-source development panel before and after each arm. Preserve all existing waveform, mel, group-boundary, quiet, near-silence and peak definitions.
- Compare absolute log RMS gain error, temporal gain variability and waveform MAE throughout the trajectory. Report speech and whistling separately. Gain 1 alone is not waveform fidelity.
- Compare the full-panel endpoint metrics, the unchanged weighted objective and the number and severity of individual regressions. Retain per-source results so a pooled average cannot hide failures.
- Verify source identity, cached teacher targets, fixed-module integrity, optimizer counters and starting state before interpreting the results.

This is a bounded diagnostic, not final quality qualification. A smoother gain trajectory with worse waveform or spectral reconstruction is a tradeoff, not a demonstrated fix. Quiet behavior is independently required in the assessment. The final 0.99 active waveform-correlation goal remains unchanged, but is not an entry requirement for interpreting an experiment.

If the comparison supports steadier updates without material reconstruction regressions, use that evidence to plan a validated continuation from the user's preserved 1,000-step anchor. Do not select the best intermediate checkpoint or silently promote either diagnostic endpoint.

If the comparison does not materially help, first consider reconstruction-aware initialization of the same-width group. If teacher-function reconstruction remains limited, use training-only evidence to choose one minimally wider internal boundary or a different factorization of the heavy matrices. Those are conditional designs, not experiments authorized or launched by this protocol.
