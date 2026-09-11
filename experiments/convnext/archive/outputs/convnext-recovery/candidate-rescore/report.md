# Saved decoder candidates on corrected inputs

**Corrected evaluation inputs did not reverse the earlier conclusions.** The later complex-discriminator candidate retains spectral benefits against its matched fresh-magnitude control, but does not beat our retained decoder on natural quiet audio. Tanh bounds peaks with a reconstruction tradeoff. The saved short-window mel candidate still makes quiet audio worse.

All seven saved decoders were rescored on the same 285 corrected crops from 147 sources. Weights, architecture, normalization buffers and the common evaluation loss stayed unchanged. No retraining, teacher generation or CPU speed benchmark occurred. Each candidate is compared with its corresponding control, at the same training step and exposure.

| Saved candidate | Natural waveform error | Common mel error | Natural quiet error | Steady encoded-zero error |
|---|---:|---:|---:|---:|
| Tanh, 200 updates | +0.49% | −0.68% | +1.66% | −3.71% |
| Short-window mel, 200 updates | +0.21% | −0.58% | +4.43% | +12.16% |
| Complex versus fresh magnitude, 200 updates | −0.09% | −0.26% | +1.92% | +6.24% |
| Complex versus fresh magnitude, later 400 updates | −0.40% | −3.51% | −3.20% | −12.50% |

Percentages are changes relative to the matched control; negative means lower error. Natural waveform error is sample-weighted MAE, mel is the unchanged equal-crop diagnostic, and quiet error is pooled residual RMS over teacher-defined 20 ms windows. Encoded zero uses seconds 2–6 of the complete six-second fixture. The complex candidates use fresh-magnitude discriminator controls, not the unchanged original discriminator.

**Tanh:** maximum peak falls from 1.299 to 0.922 and scored overshoot observations fall from 660 to zero. However, natural waveform error increases, and expressive correlation decreases from 0.75253 to 0.75021. Generic emotional/nonverbal waveform error increases 2.25%; laughter increases 0.89%. Tanh is useful for bounding output, but this checkpoint does not demonstrate quality-neutral adoption or a natural-silence fix.

**Short-window mel:** its small mel improvement still comes with worse quiet error and 12.16% worse steady encoded-zero residual. The earlier quiet penalty was 4.32%; it is now 4.43%. This candidate inherited stale gradient-balancer statistics and redistributed the mel budget across five scales. Rescoring does not remove that training confound, so this result rejects adopting the saved candidate, not shorter spectral supervision in general.

**Complex discrimination:** the original 200-update screen remains inconclusive and slightly worse on quiet audio. The later matched 400-update comparison retains its clearer benefits: high-frequency magnitude error decreases 6.74%, natural mel error decreases 3.51%, and steady encoded-zero RMS falls from 0.0001382 to 0.0001210. Expressive correlation improves from 0.77416 to 0.77578. These support retaining the method as a training candidate; the discriminator adds no decoder inference operations.

That later complex candidate still has tradeoffs. Its maximum peak increases from 1.248 to 1.263, although total peak-excess energy decreases 5.26% and scored overshoot observations decrease from 458 to 454. Laughter waveform error increases 0.62%; German increases 2.91% across four sources. Improvements are therefore not uniform across clips or groups.

**The retained decoder is a different reference:** our current `targeted` step-8,490 checkpoint kept its existing discriminator. Against its previous corrected evaluation on the identical canonical target and metric contract, the complex candidate has 2.95% lower mel error and 0.05% lower waveform error, but **1.79% higher natural quiet residual**. Thus the −3.20% quiet improvement in the table is against the fresh-magnitude control, not against our retained decoder. This practical reference was not rerun in the seven-model rescore. [Retained-checkpoint comparison](retained-checkpoint-comparison.json).

**None of these checkpoints fixes silence:** all seven still fail all 3,434 natural quiet-window checks under the existing thresholds. The tanh result demonstrates why bounded peaks and quiet residual must be assessed separately.

The earlier versus corrected paired changes are almost unchanged: tanh waveform penalty +0.50% → +0.49%; short-mel quiet penalty +4.32% → +4.43%; later complex mel benefit −3.51% → −3.51%. Absolute historical values also reflect a runtime change and, for the earlier screen, one added heldout crop. These comparisons establish consistency of the observed tradeoffs, not an isolated causal estimate of the encoder correction. Old training inputs and loss-calibration limitations remain in the saved weights.

Keep complex discrimination as a candidate for further controlled training, while retaining the current checkpoint. Keep tanh as a separate bounding choice with its measured tradeoff, and defer the short-mel recipe until its weighting is correctly calibrated. This rescore does not justify a replacement checkpoint or a new combined recipe.

[Matched comparisons and all groups](summary.json) · [Independent delta verification](comparison-verification.json) · [Source receipts](source-receipts.json)

This is a reconstruction diagnostic, not PESQ, STOI, UTMOS, DNSMOS, listening quality or CPU RTF evidence. Source groups are small; the expressive aggregate includes 21 explicitly classified sources. Overshoot counts are scored observations and may include overlapping crop samples.
