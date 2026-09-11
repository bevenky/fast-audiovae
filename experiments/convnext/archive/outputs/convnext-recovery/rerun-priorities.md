**Which earlier tests to repeat after the backend correction**

Keep the targeted step-8,490 checkpoint. The corrected 147-source panel gives nonquiet correlation 0.92011 versus 0.91991 historically, with essentially unchanged quiet residual and maximum peak. This rules against the encoder bug being the main explanation for those remaining artifacts on this panel. It does not establish which historical training inputs were affected.

| Priority | Repeat | Question it answers |
|---|---|---|
| 1 | Fresh encoder checks on authentic training sources, separately from decoding the saved latents and checking their cached targets | Did the encoder distort the training distribution, and are the conditional teacher targets mathematically correct? These are distinct failure modes. |
| 2 | The fixed-batch loss-gradient and update-dose diagnostic using corrected inputs and targets | Does a normal update still worsen quiet error, while a smaller update improves it? Are the waveform, mel, adversarial and feature-matching contributions still consistent with the earlier diagnosis? |
| 3 | Matched 400-step current-rate and quarter-rate continuations | Does the smaller update improve quiet audio persistently without worsening speech, expression, peaks or mel quality? Use the same corrected data and starting optimizer state in both arms. |
| 4 | Rescore saved tanh, short-window mel and spectral-discriminator candidates against their matched controls on corrected inputs | Did the earlier conclusions depend on affected examples? Rescore existing weights before deciding whether any candidate deserves a new training experiment. |

For each data-sensitive comparison, report the eight corrected heldout sources separately, with laughter/transients, natural quiet regions and the three encoded silence/noise/fade fixtures visible. Keep original crop boundaries, normalization, masks and teacher targets fixed within each comparison. Use ordinary speech as a regression control. A global correlation average alone is insufficient.

The new software already passed the isolated FP64 projection comparison. Full historical batch encoding matches singleton encoding bitwise with cuDNN 9.25.1. The bounded student/discriminator forward and gradient audit did not show the same defect. Repeating all kernel timing, head-rank or structural causality analyses is not the first priority because their underlying weights or structure did not change.

The previous quarter-update result came from applying 25% of one recorded parameter displacement, not from training at a quarter learning rate. Its targeted quiet squared error improved 5.23%, while peak-excess energy still worsened 4.10%. The full displacement worsened quiet error 34.96%. The small-update experiment therefore tests a potential silence improvement, not a complete overshoot solution.

The remaining optimization explanation is a hypothesis, not a conclusion. The first decisive test is whether its update-size result reproduces on corrected teacher data. Do not change normalization, architecture and loss weights together, and do not discard the existing checkpoint solely because some earlier encoded targets were wrong.

The short updated-runtime throughput comparison is separate from these quality experiments. It used disposable updates and retained no new training checkpoint.

**Completed cache audit and limits**

The conditional teacher scan covered all 17,613 distinct crops in the audited cache, including 17,328 training crops and 285 held-out crops. The teacher was frozen and historical files were unchanged. It decoded saved latents; it did not regenerate every historical training latent from source audio. Eighty-one crops exceeded the predeclared strict numerical screen (maximum absolute error 0.00001 or per-crop RMS 0.000001). The worst absolute difference was 0.0000247322, the worst crop RMS was 0.00000123580, and pooled RMS was 0.00000010348. There were no nonfinite outputs. All 93 batch-versus-singleton shape sentinel checks passed. These discrepancies are small, but the audit remains marked as failing its strict screen. They do not resemble the order-one encoder defect.

A separate fresh encoder sample checked 32 authentic training sources: 31 latent crops matched bitwise and one had a maximum difference of 0.0000268221, above its strict 0.00002 absolute screen, with RMS 0.00000165328. All 32 conditional waveform checks passed. This bounded sample cannot establish the affected fraction of all historical training inputs.

For the next loss-gradient diagnostic, regenerate both encoder latents and conditional decoder targets for the selected diagnostic batches using the corrected runtime. Verify batch versus singleton behavior on those actual batches. Preserve both the original checkpoint and optimizer state. This avoids interpreting an old-cache discrepancy as an optimizer effect. Do not mark the old training cache as having passed every numerical check.

No 400-step continuation or new architectural training experiment has started as part of these audits.
