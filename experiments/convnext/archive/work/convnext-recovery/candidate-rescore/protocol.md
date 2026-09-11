# Saved candidate rescore

Rescore seven preserved decoder checkpoints, with no retraining, teacher generation, discriminator forward or optimizer construction. All seven checkpoint files and their original completion/migration receipts were found on Runpod. `plan.json` pins their exact hashes and the corrected panel receipt.

| Candidate | Matched control | Saved step | Previous training exposure |
|---|---|---:|---|
| Terminal tanh | Original unchanged control | 8,290 | 200 generator updates each |
| Short-window mel | Original unchanged control | 8,290 | 200 generator updates each |
| Complex spectral discriminator | Fresh magnitude discriminator | 8,290 | 200 generator and 20 discriminator-only updates each |
| Later complex spectral discriminator | Later fresh magnitude discriminator | 8,490 | 400 generator and 64 discriminator-only updates each; matched targeted data and revised calibration |

The first three comparisons use `/workspace/fast-audiovae-convnext-20260909-r9/remediation/fusion-screen/<arm>/final.pt`. The last uses `corrected-screen/targeted_complex/final.pt` versus `corrected-screen/targeted_magnitude/final.pt`. All descend independently from the same preserved step-8,090 parent. The later complex candidate must not be compared against the 200-update magnitude control or against the original unchanged discriminator.

Use the sealed canonical panel at `/tmp/fast-audiovae-recovery-20260909/canonical-panel-v1/receipt.json`: all 285 crops, 147 sources, identical masks and teacher-defined quiet windows. The earlier candidate panel contained 284 crops; the canonical panel additionally includes the reserved Yell source. The script checks that every canonical source remains absent from the corresponding historical training selection. It does not drop difficult or overlapping sources to force completion.

Only saved student tensors and the common diagnostic criterion are restored. The tanh checkpoint restores its saved bounded head; short-mel and discriminator candidates retain their original decoder architecture. A strict reconstruction of the saved model identity verifies normalization buffers, model config, architecture flags, migration receipt and parameter hash. It also checks that current inference and diagnostic source files match the original experiment's hashes. Changed loss functions are never substituted into evaluation.

Reports retain waveform MAE, the common mel diagnostic, active correlation, natural quiet residual, steady encoded-zero residual, overshoot/peak energy, saturation, high-frequency spectral errors, and all language/event/source groups. Waveform and quiet errors use their existing sample weighting; mel retains equal-crop averaging. Undefined ratios are reported explicitly. Paired differences are descriptive, with no automatic adoption decision.

Important limits remain:

- Rescoring corrects evaluation inputs and targets. It does not repair the latent distribution or targets used to train these old checkpoints.
- The original short-mel and complex-discriminator candidates inherited stale gradient-balancer statistics. Their 200-update results cannot establish the methods' best achievable quality.
- The later complex comparison has its own matched corrected-training control. It is the more relevant existing discriminator comparison, but still a short saved experiment.
- Changes from historical reported numbers combine a newer numerical runtime, corrected teacher targets and, for the earlier screen, one additional crop. Compare candidates against their freshly rescored controls rather than attributing historical deltas to one cause.
- No PESQ, UTMOS, DNSMOS, listening study or CPU RTF test is included in this reconstruction rescore.

Root schedules the GPU. With the qualified environment and existing canonical helper modules on `PYTHONPATH`, run:

```sh
python rescore_candidates.py --out /tmp/fast-audiovae-recovery-20260909/candidate-rescore-v1
```

Use `--pairs terminal_tanh_200 short_mel_200` for the first three unique decoders, or `--pairs complex_spectral_200 complex_spectral_corrected400` for the two discriminator comparisons. Every invocation needs a new output directory; existing evidence is never overwritten. `summary.json` records the four matched comparisons, and individual files retain every scored crop.
