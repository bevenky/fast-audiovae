# Corrected-input update diagnostic

The corrected teacher inputs reproduce the important earlier finding: the full native update worsens quiet reconstruction, while a quarter-rate update improves it on this panel. This supports the already approved controlled 400-step comparison. It does not establish the best learning rate or justify replacing the optimizer, losses or architecture.

Both trials started from the same retained step-8,490 checkpoint, optimizer moments and loss EMA. They used the same 32 freshly regenerated training crops and 18 canonical held-out probes. The teacher encoded authenticated complete sources using the singleton cuDNN-disabled reference path before targets were cropped. The student ran under Torch 2.14 with cuDNN 9.25.1. No checkpoint changes were retained.

| After one actual native D-then-G step | Current rate | Quarter rate |
|---|---:|---:|
| Quiet residual MSE | +35.134% | −5.224% |
| Raw waveform MAE | +0.432% | −0.095% |
| Waveform MSE | +0.736% | +0.021% |
| Teacher-relative peak-excess energy | +17.536% | +4.101% |
| Samples exceeding full scale | 266 → 287 | 266 → 271 |

Percentages compare equal-crop diagnostic means against the starting checkpoint. Quiet MSE is squared error, not noise amplitude, and is not the sample-pooled quiet RMS used by the larger quality screen. Six probes contained no qualifying quiet samples. The two actual-rate trials had identical pre-update losses and gradient norms; only the joint generator rate differed. The actual quarter-rate result closely matches the earlier quarter-length parameter-path result, which had only suggested this experiment.

The benefit is not confined to synthetic silence. At the quarter rate, quiet error improves on 10 of the 12 probes containing quiet samples and raw MAE improves on 13 of 18 probes. Natural-only mean quiet MSE improves by 3.619%; speech and expressive subsets improve by 12.001% and 2.505%, respectively. These small selected subsets are diagnostic examples, not language-wide or event-wide estimates.

Two quiet outliers remain: the screaming probe worsens by 2.310% and the breathing probe by 5.582% at the quarter rate, although both worsen much more at the current rate. Pure encoded-zero MSE falls by 60.410%, but its maximum amplitude rises from 0.000623 to 0.000741. Thus lower average error does not mean every transient or startup maximum improves. Peak-excess energy and overshoot counts still increase on the natural probes. Smaller updates help stability; they have not solved peak control.

The corrected loss diagnostic still rewards approaching the teacher. With fixed checkpoint-derived scales and identical discriminator views, the weighted objective decreases across the five artificial output blends: **59.475 → 30.147 → 6.271 → 0.656 → 0.00243**. These blends are a diagnostic interpolation, not successive trained checkpoints or measured correlation targets. Before the exact-teacher endpoint, the combined directional derivative toward the teacher is negative for active audio, quiet audio, original overshoots and high teacher peaks. Waveform, mel and feature-matching losses and their recorded output gradients become exactly zero when student output is replaced by the teacher output.

The fixed discriminator still gives a nonzero adversarial loss and gradient at the exact teacher. This alone does not prove an incompatible optimum: the reconstruction terms have nondifferentiable points there, and the sampled output path does not establish every reachable parameter direction. There is no evidence here to remove GAN or feature matching. Nor should zero adversarial gradient on a particular overshoot region be generalized beyond the selected short discriminator views.

The first-order prediction for the full update still suggests decreasing quiet error, while the actual finite update increases it. Along the fixed update direction, 10%, 25% and 50% displacement improve quiet MSE, and 100% worsens it. The evidence therefore still points to finite-update size and interactions on this batch, rather than the encoder-cache defect explaining away the finding. Partial-parameter counterfactuals remain nonadditive and do not establish that Muon or AdamW should be removed.

Proceed with the two matched 400-step arms using the separately sealed fresh 12,800-pair overlay. Keep the current recipe, discriminator policy, normalization and initial optimizer/EMA state identical. Judge both arms on the complete historical and canonical panels with their unchanged gates, including individual source regressions and peak behavior. The one-step result is sufficient to justify that controlled test, not to select the quarter rate permanently.

Evidence: [corrected-update-diagnostic.json](corrected-update-diagnostic.json), [fresh teacher generation](fresh32-reference-generation.json), [fresh training receipt](fresh32-reference-receipt.json). Original checkpoint hashes and fresh overlay hashes passed preservation checks; all disposable engine states were restored exactly.
