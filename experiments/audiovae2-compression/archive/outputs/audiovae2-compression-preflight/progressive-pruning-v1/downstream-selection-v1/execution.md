# Downstream-aware selection plus B

Fresh original teacher initialization completed on11 September2026. No trained student checkpoint was installed. All nine residual units and widths384/256/128 are preserved. Four existing native operators were fitted using the unchanged B algorithm after selecting a new support from downstream reconstruction statistics.

- Frozen code archive SHA256: `8754314b59a31ecc178eaeff9c6d1afe28cee9cd0ebd8c42277e251f1b1a86ca`. Local39 tests and qualified remote39 CPU tests passed.
- Native operator artifact SHA256: `10c9f147c3cf14fa5a07f7724c46a629f5b471a0d45f2407f6b92246a959fe43`.
- Complete initial decoder state SHA256: `d9c379862c17de6a368284bd531976df95ce02595ddad71ccf08f845f053a669`.
- Initializer: `/workspace/fast-audiovae-downstream-20260911-v1/init`;456 teacher-cache checks passed;35.22seconds;72 calibration and96 development sources.
- Recovery: `/tmp/fast-audiovae-downstream-recovery-20260911-v1/results`; PID1099726 launched with an exclusive lock. Source remains frozen under the initializer root. A separate filesystem provides the checkpoint reserve without deleting earlier artifacts.
- Initial recovery parity passed for quality, full native state, original RNG and empty Adam. All90 group tensors train, original teacher/outer stages remain frozen, physical batch1/accumulation12.
- Exactly2,000 updates and24,000 distinct sources, matched to B; original objective/Adam recipe; reviews0/250/500/1000/1500/2000. No quality-based early stopping, extra training loss or automatic continuation.
- TensorBoard port8888 now serves this candidate. Previous C events remain untouched.

Initializer metrics are in [aggregate evidence](initializer-aggregate.json). Waveform MAE0.00445364 is32.75% below B's initialization; startup11/13 pass versus0/13, but overall quiet passes628/2544 versus751/2544. These are step0 results, not a trained-quality conclusion.

The first execution failed to save its final checkpoint after all2,000updates and reviews. See [storage failure and fresh repeat](storage-failure.md). Its failed receipts/results remain unchanged. The identical fresh repeat has now completed in an isolated, capacity-checked location and passed its saved-checkpoint audit. All2,000loss records and six saved quality reviews exactly reproduce the first attempt. See [completed findings](completed-findings.md) and [aggregate evidence](completed-repeat-aggregate.json). No CPU RTF or perceptual qualification is implied.
