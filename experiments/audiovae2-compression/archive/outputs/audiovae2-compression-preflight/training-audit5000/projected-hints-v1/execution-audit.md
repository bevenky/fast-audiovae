The fixed comparison completed in 252.38 seconds after authentication/model loading, with no execution failure. Both arms restored the same retained accumulation12 checkpoint at optimizer counter4625 and consumed the same ordered1,500 diagnostic sources in125 singleton-accumulation12 updates, ending at4750. Reusing this already-seen interval was explicitly allowed for the controlled diagnostic. It is not fresh-data continuation.

All12 focused CPU tests passed on Runpod. They include exact baseline parameter and AdamW-moment equality against the unchanged original update, both with and without diagnostic backward measurements. In the real comparison, the complete96-source initial reports were exactly equal between arms, and both passed the saved checkpoint's existing tolerance. All3,000 training teacher/cache checks were bitwise equal. Original4500/5000 and retained candidate files, frozen student modules and teacher remained unchanged. Main training remains paused.

The two affine projectors were ridge-initialized using only the72 calibration sources, then jointly trained with a separate fresh AdamW optimizer. Student AdamW moments and the three original coefficients were retained. Each hint received a fixed5% norm budget on its connected student parameters; their combined all-student gradient norm was6.8185% of the existing objective, cosine+0.06997 at calibration. Readouts remain outside the decoder, and its state keys and deployed shapes remain unchanged. Actual CPU streaming/export validation would still be required before adoption.

| Metric | Baseline | Two hints | Change |
| --- | ---: | ---: | ---: |
| Waveform MAE | 0.00417039103 | 0.00417764102 | +0.174% |
| Mel | 0.329309583 | 0.330991149 | +0.511% |
| Full group MSE | 0.0113457317 | 0.0108877887 | -4.036% |
| Quiet residual RMS | 0.00017437838 | 0.000173869493 | -0.292% |

At the fixed96-source endpoint, quiet-window failures were2379→2394 out of2544; near-silence failures181→183 out of184. Both arms had zero full-scale overshoot samples. Therefore the lower internal MSE does not qualify this hint configuration as an overall waveform-quality improvement.

The hint arm also improved feature-prediction losses when evaluated through the original frozen readouts, not just its trained readouts. That establishes a change in the student's intermediate information, but does not prove that information is useful for the frozen downstream waveform decoder. The complete128-channel group output remained directly supervised throughout.

Only first and final actual training updates included branch-gradient dot actual-student-displacement diagnostics; there was no learning-rate search or disposable update. Full96-source endpoints and26 equally spaced four-case observations were fixed in advance. No intermediate best checkpoint was selected. All result JSON/JSONL and logs were retrieved; large group/optimizer/projector checkpoints remain on Runpod. No production changes, commits or automatic promotion occurred.
