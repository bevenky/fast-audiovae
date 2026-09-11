# Startup-preserving initializer result

The bounded constrained-A experiment completed successfully on Runpod in30.670 seconds of core execution. It used no neural optimizer updates and introduced no new inference operations.

| Metric | Original A | Startup-constrained A |
|---|---:|---:|
| Active waveform cosine |0.953530|0.949782|
| Waveform MAE |0.007477|0.007858|
| Mel error |0.268096|0.279047|
| Group MSE |0.003265|0.003532|
| Quiet residual RMS, microFS |166.773|181.733|
| Startup residual RMS, microFS |12.272|2.088|
| Startup passing |0/13|13/13|
| Other near-silence passing |166/171|167/171|
| Remaining quiet passing |336/2360|206/2360|
| All quiet passing |502/2544|386/2544|

microFS means one millionth of full scale. Correlation is not perceptual accuracy. The quiet groups shown above are disjoint. The thirteen startup cases pass existing waveform thresholds, not exact waveform equality.

Compared with A, waveform MAE worsens on92/96 recordings, mel and group errors worsen on96/96, and cosine worsens on88/94 eligible recordings. The reduction in total quiet passes is a net116 windows. Therefore this is a useful diagnostic success, not an overall improved model or a promoted candidate.

The720 native constraints come from6 calibration sources and5760 waveform samples, separate from the96 development sources. The fixed numerical policy retained360 constraint modes and discarded360. This rank is not a measure of overall model capacity. All6 installed FP32 calibration equality checks passed; maximum absolute native constraint error1.19209e-6. The native operator writeback checks passed too.

The same384-channel model can produce a passing startup response for this observed panel. This argues against restoring width or adding a layer being necessary for these specific failures. It does not establish universal startup coverage or that subsequent unconstrained training will preserve the result.

The changed upsampler weights are shared over the entire recording. Improving its startup mapping changes ordinary audio too. Moreover, exact hidden-interface matching is stricter than the output waveform requirement. The measured tradeoff should not be hidden by reporting only the13 passes.

Keep this candidate as evidence. Do not copy its upsampler into B or the trained5,000-step model, because their actual upstream inputs differ. The B recovery run remains unchanged and is evaluated independently.

## Execution provenance

- Remote root:/tmp/fast-audiovae-progressive-pruning-v1/startup-constrained-init-v1
- PID1085345 completed; status evaluated.
- Original baseline reproduced; protected files, teacher and original step0 state preserved.
- Local and qualified Runpod CPU tests:8 passed.
- Source freeze SHA256:f9a33b41aeabf163d145f8adc10c7290ac901e3a0a6c221025f21bacb6bd8f31
- Source SHA256:9c189dcec82718f47e81c374ee484415c893d58fd05a2486d68cf2dec55808ac
- Test SHA256:36a7350ed5a0f14764d0630230f64f48c592c2a9aa1d9c5aee4cc4e392b7b649
- Qualified test log SHA256:52580c0cb2469187183288c875a82f8f582cfd9305ea41f29060b31c31816480

The original fixed ridge and numerical tolerance were not tuned against development. All raw waveforms, latent arrays, source identities, individual results and fitted native tensors remain on Runpod. The local result files contain aggregates only. No commits, push or automatic promotion occurred.

