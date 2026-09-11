# Independent contribution and reconstructibility audit

These measurements show that the first removed contribution is substantially predictable from retained features, but repairing that operation alone does not repair the decoder. Even supplying its exact missing contribution has almost no benefit to total waveform error while worsening quiet reconstruction. Restoring the measured contributions at all eight affected mixing operations recovers teacher output to numerical precision on the targeted panel.

The scope is the **original untrained sliced initialization**, not the adapted 1,000–5,000 checkpoints. The trained accumulation-12 model was not modified or retested. Values below must not be compared as though this untrained model replaced that checkpoint.

## Integrity and numerical controls

- The affine map uses only the 72 fixed calibration sources. All 96 development sources are disjoint, retain their exact ordered identities, and were evaluated after freezing the map.
- Across plain, exact-first and fitted candidates, all per-source sample counts, active/quiet support, teacher RMS, group target energy and mel element counts match. The three teacher ablations and all-eight oracle use the same 15 targeted sources.
- Every recorded live teacher target is bitwise equal to its cached target: 420 source evaluations across the eight reports. The original checkpoint files, teacher, frozen student sections and original sliced group are preserved.
- The same-width copied decoder is bitwise equal to the teacher on all 15 controls. The narrower first-mixer input has small FP32 drift: worst input RMS error 2.512e-7, worst normalized RMS drift 9.506e-7. The corresponding retained-contribution drift is recorded separately, with maximum RMS 1.641e-7.
- The affine solve uses weighted centered FP64 statistics, an unpenalized intercept and the fixed ridge. Effective rank is 256/256, regularized condition approximately 2,752.5, and normal-equation relative residual 8.47e-16. Explicit versus folded correction passes with maximum difference 1.43e-6. No near-zero input normalization or heldout tuning is used.

The initial absolute-only alignment gate stopped a previous diagnostic attempt. The successful v2 checks actual input values with the existing operator tolerance and records the drift. No waveform-quality threshold changed.

## Can retained features reconstruct the first missing contribution?

The comparison is against leaving the missing contribution at zero. Squared errors and target energy are pooled using valid waveform-sample weights at each feature cell; short tails remain included. These are **local hidden-contribution scores**, not waveform-quality percentages.

| Region | Fit sources with support | Fit SSE reduction | Heldout sources with support | Heldout SSE reduction |
|---|---:|---:|---:|---:|
| All valid audio |72|76.67%|96|75.86%|
| Active audio |70|75.52%|94|74.08%|
| Quiet audio |52|80.78%|68|82.00%|
| Near-silence |8|93.18%|15|93.32%|
| Scored startup, first 40 ms |29|69.62%|42|70.89%|

All 96 heldout sources improve this local reconstruction. Their individual whole-source SSE reductions range from 65.04% to 87.36%. Pooled heldout omitted-contribution RMS is 0.134113 and fit residual RMS is 0.065886, so meaningful error remains despite the improvement. Near-silence pooled target RMS is 0.244303 and fit residual RMS is 0.063146. These hidden values are not acoustic noise levels.

For the whistle, local whole-source SSE improves 77.66%, but final waveform MAE becomes slightly worse. This is direct evidence that a successful local regression does not establish a successful decoder repair.

## What happens to final audio on the same 96 development sources?

| Metric | Original sliced initialization | Exact first contribution restored | First contribution fitted and folded |
|---|---:|---:|---:|
| Waveform MAE |0.02480914|0.02480411|0.02481748|
| Waveform MSE |0.00351161|0.00350831|0.00351089|
| Common mel error |1.674659|1.649386|1.656917|
| Stage-4 group MSE |0.0960262|0.0958836|0.0959688|
| Active correlation |32.7810%|31.6261%|30.9031%|
| Quiet residual RMS |0.000970420|0.001046455|0.001026707|
| Quiet windows failing |2544/2544|2544/2544|2544/2544|

Exact first restoration lowers MAE only 0.0203%, while the fitted correction increases it 0.0336%. Quiet residual RMS worsens 7.84% and 5.80%, respectively. These changes are not a practical decoder-quality win.

Per-source results also reject a blanket improvement claim:

| Comparison with sliced initialization | Exact first restoration | Fitted first restoration |
|---|---:|---:|
| MAE improves / worsens |57 / 39|35 / 61|
| MSE improves / worsens |77 / 19|63 / 33|
| Mel improves / worsens |73 / 23|69 / 27|
| Group MSE improves / worsens |61 / 35|50 / 46|
| Active correlation improves / worsens |28 / 66|12 / 82|
| Quiet error improves / worsens, supported sources only |6 / 62|9 / 59|

The worst fitted MAE increase is the Lithuanian quiet crop, from 0.000703010 to 0.000824078. Armenian and Azerbaijani quiet crops also worsen. The unchanged source and region counts rule out improved coverage being the reason for these comparisons.

## Direct interventions into the original teacher

Each ablation removes only the specified contribution into the selected output coordinates, retaining other teacher channels downstream. These are counterfactual tests of contribution importance, not candidate decoders.

| Intervention on the same 15 sources | Waveform MAE | Active correlation | Quiet failures |
|---|---:|---:|---:|
| Remove first stage-2 residual mixer contribution |0.00458429|98.2705%|401/595|
| Remove stage-3 upsampler contribution |0.01488139|95.1288%|595/595|
| Remove stage-4 upsampler contribution |0.02879017|89.5935%|595/595|
| Supply all eight missing contributions to sliced model |1.323e-8|approximately 100%|0/595|

The all-eight oracle also has zero failures across 122 near-silence windows and quiet residual RMS 1.190e-9. Its stage-4 MSE is 3.056e-14. This strongly supports the completeness of the missing-contribution accounting: it recovers the teacher despite the much larger ordinary sliced-model error.

The ablations have different output-coordinate scopes, so their magnitudes do not constitute an intrinsic ranking of layer importance. Restoring all eight operations also uses external teacher features. It is an accounting control, not proof that the narrow student can synthesize those contributions unaided or a deployable CPU solution.

## Decision

Do not adopt the isolated first-mixer folded correction as a repair, and do not patch it into the adapted checkpoint. It predicts the local missing signal well on heldout data, but the whole decoder gains no meaningful waveform fidelity; an exact local oracle fares similarly. The remaining omissions and their interactions are therefore material, and the first mixer is not a sufficient repair point.

The results support investigating reconstruction across the affected group rather than adding a special silence or whistle layer based on this one operation. They do not yet establish an optimal narrower architecture, a capacity floor, or that a broader reconstruction initializer will outperform the already trained model. Any such next step remains a separate proposed experiment.

Exact sufficient-statistic reductions, complete per-source differences, integrity checks and hashes are in [independent-contribution-audit.json](results/independent-contribution-audit.json). Original reports and waveform snippets remain unchanged in [results](results/).
