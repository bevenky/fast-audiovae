# Student and discriminator backend audit

The encoder defect has **not reproduced in the student or discriminator** in this bounded check of the saved step-8490 student. Within each backend, changing batch size from 32 to 1 or 8, repeating the call, and returning after a different input length produced bitwise-identical matching outputs. Comparing cuDNN on against off produced student waveform differences of about 0.00011% relative L2 and at most 0.0000023 in amplitude.

Gradients were not bitwise identical, but the complete parameter-gradient differences were small:

| Quantity | Relative L2 difference, cuDNN on versus off |
|---|---:|
| Student gradients with identical external waveform gradient | 0.05122% |
| Student gradients with the full balanced training objective | 0.01656% |
| Discriminator gradients with identical real and synthetic-fake inputs | 0.0005918% |

These percentages pool all named parameter tensors. The largest individual student tensor differed by 0.09357% for the external-gradient test and 0.13523% for the balanced objective. The balanced waveform gradient differed by 0.39720%; the corresponding parameter gradients differed considerably less. These are numerical differences, not measured quality losses or percentages of incorrect training.

The waveform loss was identical in both runs. The other losses differed by approximately 0.000000015 to 0.0000012. The two runs also produced almost identical loss-balancer scales. All complete engine-state hashes, including optimizer state, loss EMA and crop RNG, were unchanged after the audit. No optimizer step was taken.

Small forward differences can produce larger gradient differences near the derivative discontinuities of PReLU, LeakyReLU and absolute-error losses. The current result is compatible with that explanation, but it does not directly establish how much each mechanism contributes. The external-gradient test excludes reconstruction losses and the discriminator, so its remaining difference cannot be attributed to those losses alone.

This supports preserving the trained checkpoint while fixing and measuring the encoder/cache mismatch. It does not certify every historical checkpoint, input length or call order, and it does not prove that all earlier training used production-consistent encoder latents. Nor does it establish that the observed gradient differences would remain small through thousands of optimizer updates. The tested generator objective used the saved discriminator weights, without the discriminator update that precedes an ordinary generator step.

If a further numerical investigation becomes necessary, the smallest useful probe is an isolated backward comparison for the student head convolution using exactly the same saved input, weights and upstream gradient under both backends, with a float64 reference. Separately recording how often the preceding PReLU input changes sign would distinguish backward-kernel arithmetic from sensitivity to tiny forward changes. This is a diagnostic proposal, not a new training change or a prerequisite established by these results.

Evidence: `work/convnext-recovery/backend_training_audit/comparison.json`, Torch 2.11.0+cu128, cuDNN 9.19, H100 NVL, targeted checkpoint step 8490, one fixed batch of 32 cached training crops. Student and discriminator forward features, gradients and all raw norms were recorded separately.
