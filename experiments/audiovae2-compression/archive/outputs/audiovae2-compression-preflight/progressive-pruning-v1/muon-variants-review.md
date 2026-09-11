# Optimizer comparison for pruned AudioVAE2 recovery

11 September 2026. Research and source review only. The current 2,000-update AdamW run is unchanged. None of these optimizer trials has started.

## Recommendation

Compare AdamW, selective Muon plus AdamW, selective NorMuon plus AdamW, and selective Shampoo plus AdamW from the same fresh G-plus-startup initializer. Plain Muon establishes the effect of the matrix optimizer; NorMuon then isolates the extra adaptation. Shampoo supplies a distinct matrix-preconditioning comparison with published convolutional-network evidence. No candidate is established as better for this decoder. Start with a short matched qualification before considering a full independent run.

The user requested the Shampoo addition after the original three-arm recommendation. The existing Muon review below is retained; the convolution review and expanded plan follow it.

## Required follow-up after all four optimizer comparisons

The user explicitly deferred the quiet-window regression investigation until after AdamW, Muon plus AdamW, NorMuon plus AdamW, and Shampoo plus AdamW have completed their comparisons. Do not interrupt the current run or start that investigation early. Keep the existing milestone measurements and preserve their underlying evidence during the comparisons.

The current run's fixed development panel declined from 977/2,544 quiet windows passing at step500 to 856/2,544 at step1000, a net loss of121 passes (38.40% to33.65%). Average quiet residual RMS nevertheless improved from100.073 to88.802 millionths of full scale. Neither continuous average improvement nor increasing active correlation proves quiet quality is improving; pass counts need not be monotonic under this training objective. This observation is not yet a root-cause diagnosis.

After the optimizer comparisons:

- Verify the same teacher targets, window identities, sample alignment, masks, cohort membership and thresholds across arms and available saved steps, including initialization. Compare matched ordinary-source exposure and report wall time separately.
- Count pass-to-fail, fail-to-pass, persistent-pass and persistent-fail transitions. Separate residual-only, amplitude-only and combined failures, with aggregate error distributions and margins from the unchanged limits. Do not infer individual transitions from net counts.
- Report all seven overlapping quiet cohorts separately, distinguishing protected calibration startup from development startup and ordinary quiet audio. Do not sum overlapping cohorts.
- Check whether the decline is consistent across optimizers and whether quiet improvements persist through the final review. Do not select an optimizer solely by active correlation or average waveform error.
- Use existing saved evidence first. If intermediate model states were not saved, explicitly mark them unavailable instead of claiming an exact replay. Any subsequent targeted diagnostic should answer the gaps remaining after the four comparisons.

Only aggregate counts and distributions may leave Runpod. Keep per-window values and identifiers there. No thresholds, loss weights, architecture, optimizer or protection policy are changed by this follow-up entry.

| Method | Addition | Decision for this decoder |
|---|---|---|
| [NorMuon](https://arxiv.org/html/2510.05491v1#S3.SS1) | Adaptive post-orthogonalization scaling per output neuron | First additional comparator. The paper reports 11.31% better training efficiency than Muon in a 1.1B language-model setting, not an audio result. |
| [AdaMuon](https://arxiv.org/abs/2507.11005) | Per-coefficient second moments and sign-stabilized orthogonalization | Second wave. More adaptive freedom, but more state and a larger algorithm change. A library class bearing the name must match the intended paper version. |
| [Polar Express](https://arxiv.org/abs/2505.16932) | Improved polynomial approximation of the polar factor | A possible common numerical implementation for both Muon arms. Do not change it in only one arm while claiming to isolate NorMuon. [Author implementation](https://github.com/NoahAmsel/PolarExpress). |
| [Newton-Muon](https://arxiv.org/abs/2604.01472) | Input second-moment preconditioning before Muon | Relevant if poor input conditioning remains limiting, but requires activation statistics and inverse updates. Defer until the simpler comparison. [Author code](https://github.com/zhehangdu/Newton-Muon). |
| [Muown](https://arxiv.org/abs/2605.10797) | Explicit row-magnitude and direction optimization | Interesting for norm drift; our existing magnitude/direction parameterization makes this a separate integration experiment, not a drop-in addition. |
| [Dion2 / Dion3](https://github.com/microsoft/dion) | Partial-matrix orthogonalization and, for Dion3, NorMuon-style adaptation | Lower priority for nine small matrices on one GPU. Communication savings do not apply to this setup; total-time benefit is unmeasured. |
| [NuMuon](https://arxiv.org/abs/2603.03597) | Nuclear-norm constraints intended to improve later compressibility | Future compression research. Low-rank compressibility is different from restoring accuracy after our current channel cut. |
| [MuonClip](https://arxiv.org/html/2507.20534) | Attention query/key stabilization | Skip here: the pruned group has no attention query/key projections. It is not an audio peak limiter. |

## Exact candidate scope

The nine residual pointwise direction tensors are `model.{3,4,5}.block.{2,3,4}.block.3.weight_v`, three each at 384, 256 and 128 channels. Treat each as an independent C-by-C matrix, with output channels as rows. Keep the other 81 native trainable tensors on the existing AdamW recipe: weight-normalization gains, biases, Snake parameters, conditioning embeddings, depthwise filters and transposed upsamplers. Teacher, encoder and outer decoder portions remain frozen.

Do not select parameters using dimensionality alone. Several gains and activations have singleton dimensions, and embeddings are also matrices.

## Why the benefit is uncertain

For one weight-normalized row, write v = r u and W = g u. Its first-order change is:

`delta W = u delta g + (g/r) (I - u u^T) delta v`.

NorMuon's row scaling changes the directional term, so it is not canceled by weight normalization. But it does not account for the gain/radius factor or guarantee balanced changes in effective convolution weights. This is our mathematical application to the model, not a published codec finding.

The proposed matrices are square. Exact full-rank polar factors already have equal row norms; the NorMuon paper discusses this limiting case. Any additional benefit here depends on approximate orthogonalization and optimization history. That imbalance has not yet been measured in this student. No guaranteed silence or amplitude improvement follows from the optimizer name.

## Released implementation caveat

Static review of current [Microsoft NorMuon](https://github.com/microsoft/dion/blob/main/dion/normuon.py) and its [batch helper](https://github.com/microsoft/dion/blob/main/dion/megabatch_base.py) found that directly enabling flattening on C-by-C-by-1 filters does not preserve the intended algorithm. The original tensor shape is restored before last-axis normalization, and the same-shape batching path also needs an independent matrix axis. This is a source-level finding, not a runtime reproduction or a defect in our current AdamW run.

Use explicit two-dimensional update/state handling. Verify one neuron statistic per output channel and independent updates for each filter matrix. Pin an implementation revision before any experiment. The [author repository](https://github.com/zichongli5/NorMuon) is another reference implementation.

## Fair qualification and retention

Use identical initializer, data order, ordinary-source budget, losses and six calibration sources. Start each arm with empty optimizer state. Give each optimizer a small, equal training-only learning-rate qualification budget; a nominal learning-rate match alone is insufficient. Hold momentum, orthogonalization implementation, precision and all other recipe changes explicit.

The current wrapper validates AdamW-specific moments and counters. A separate mixed-optimizer adapter must snapshot and restore both state types, preserve parameter identities, and capture the combined actual 90-tensor proposal. The existing 12 waveform constraints and bounded correction remain the final acceptance operation. Never normalize weights again after acceptance.

Compare time to matched quality, waveform/mel/group error, active and effective-weight amplitude, every quiet cohort, startup calibration versus development, peak overshoot, accepted movement and correction frequency. Record setup/compilation separately and include it in total experiment time. No CPU inference benefit is expected from changing the optimizer alone.

This note proposes tests. It does not launch them, extend the current run, select a winner, or authorize another pruning cut.

## Convolution-specific review and fourth comparison

**Add Shampoo plus AdamW to the initial comparison.** [Distributed Shampoo's paper](https://arxiv.org/abs/2309.06497) includes ImageNet/ResNet-50 experiments. Its gradient statistics capture relationships between matrix directions, making it a plausible way to recover after channel pruning. The connection to our recovery task is an inference; the published image-classification results are not audio-distillation results.

| Arm | Nine pointwise direction matrices | Other 81 trainable tensors |
|---|---|---|
| Control | AdamW | AdamW |
| Muon | Muon | AdamW |
| NorMuon | NorMuon | AdamW |
| Shampoo | Shampoo with explicit Adam-based update-scale grafting | AdamW |

The [maintained PyTorch implementation](https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo) supports single-device use. Register separate optimizers if necessary so that the complementary 81 tensors preserve the exact existing AdamW behavior. Grafting means using a diagonal optimizer's scale for a Shampoo direction; it does not mean applying two parameter updates to the same weight.

For this arm, verify independent C-by-C pointwise matrices after singleton-dimension handling, full matrix factors rather than accidental vector flattening, and an explicit inverse-root refresh interval. Preconditioning must actually begin within the short qualification; otherwise the pilot could measure only the grafted fallback. Record factor updates, failed/fallback roots, damping, numerical precision and actual accepted displacement. Do not silently substitute SOAP, spectral descent, schedule-free averaging or a different momentum policy through library defaults.

These are small matrices compared with the large-model benchmarks, but no overhead estimate is yet measured. Reuse the original physical batch and accumulation. Tune a small, equal training-only budget for learning rates; report extra Shampoo configuration trials and their cost separately. Compare full quality at equal source exposure and time to a matched target, including constraint-check costs. Keep the same final 12-constraint acceptance operation and both optimizer states under transaction rollback.

## Other convolution candidates considered

- **SOAP:** a useful follow-up because it applies Adam in Shampoo's learned basis. The [authors explicitly state](https://github.com/nikhilvyas/SOAP) that their original experiments were on transformers, although their implementation accepts higher-dimensional layers. Keep it in reserve rather than describing it as proven on this decoder or as the strongest CNN evidence.
- **AdaHessian:** has published convolutional results, but its [implementation](https://github.com/amirgholami/adahessian) requires higher-order autograd. That increases the training integration and derivative workload. Defer for this short recovery comparison.
- **AdaBelief:** has [CNN and image-GAN experiments](https://proceedings.neurips.cc/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf). It remains a possible inexpensive diagonal alternative; those generalization results do not establish faster exact waveform reconstruction after pruning.
- **AdaFisher:** a [released second-order method with vision experiments](https://github.com/AtlasAnalyticsLab/AdaFisher). Adapting activation/backpropagation statistics to weight-normalized Conv1d parameters adds another integration dimension; defer rather than expanding the first matrix-optimizer comparison further.
- **DASH:** a [2026 Shampoo implementation improvement](https://arxiv.org/abs/2602.02016), with [released code](https://github.com/IST-DASLab/DASH). Its batched factors and faster inverse-root solvers are relevant if Shampoo's optimizer step becomes limiting. The paper's maximum optimizer-step speedup is not an end-to-end recovery speedup. Treat it as a later implementation comparison, not a fifth optimizer arm or an assumed win.

The fourth arm is planned only. No code was installed, no GPU experiment started, and no active training setting changed during this review.
