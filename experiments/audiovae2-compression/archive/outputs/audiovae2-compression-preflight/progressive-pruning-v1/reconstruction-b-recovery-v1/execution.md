# B recovery execution

Status: completed the authorized 2,000 updates and stopped for review. No next cut or further training was started by this run.

The approved matched recovery run launched on Runpod from the original teacher-derived step0 plus the sealed B reconstruction operators. No adapted training checkpoint initializes it.

- Remote root: /tmp/fast-audiovae-progressive-pruning-v1/reconstruction-b-recovery-v1
- Trainer PID:1085620.
- TensorBoard PID:1085845, port8888, isolated new event directory. Old event files remain unchanged.
- Maximum updates:2000. Reviews 0/250/500/1000/1500/2000; saved model/Adam/RNG/source ledger at0/1000/2000.
- All 24,000 required sources are ready, including70 sealed fresh-target shards. Their order is authenticated against the original executed journals and2,000-step ledger.
- Original teacher/encoder and student outer stages remain frozen. All 90 stage2–4 tensors train jointly.
- Initial B candidate state,96-source quality, fresh Adam and original RNG parity all passed before update1.
- Local and qualified Runpod CPU tests:20 passed.
- Source freeze SHA256:52b2d3571f5b85d3b00e80406c19a19e8517ad5f29a85f3f837e7eac2e5ac04f
- Runner SHA256:e4e66a190b63b20c836278d9893f4cd36d18b427a21da032ab3af13f672f83cf
- Monitor SHA256:05d99f051723bf0907f4e1bf60eaf4220e5305fa4efd0122eeb677e3bb2b35fb
- Tests SHA256:2f16687f96867ce85cffe6df0330d7e5658bdd3ffa1ea71a96dc957e0b6d67c4
- Remote test log SHA256:62d282b9acf76e9fe24c5a32a91696d2730403292d227e714054ee239aac0610
- B operators SHA256:7281436098cf01ac3aee09056651bdad4b7ddc112b82a22ad380edc2073866d7
- Starting decoder state SHA256:b6e8950402780c72951ccbae4da009d99e27e9bc0c1cf7ea2a161569d8c9cf22

The separate constrained-A diagnostic completed before this GPU run. Its weights or constraints are not incorporated in B training. Original checkpoint files remain protected. Only aggregate progress, metrics and provenance are returned from Runpod.

## Completion

- All 24,000 fitting sources were distinct, used once in the original order, with 16.33355349 hours of scored audio.
- All 24,000 teacher-cache comparisons passed the original tolerance. No new tolerance was introduced.
- All 90 Adam state counters reached 2,000; moments and trained weights were finite. Optimizer settings and parameter order remained unchanged, and checkpoint identity matched the launch record.
- Frozen teacher/student tensors and all protected original files were preserved.
- Reviews 0/250/500/1000/1500/2000 completed. The saved quiet-window captures reproduce each report's region summaries and retain the same window identities. The old 5,000 comparison is bound to its protected checkpoint receipt.
- Final checkpoint SHA256:43ef31f30cca751167cf91152f3ab039592b06860f0ffbd60531c8862c783954.
- Training-update time:1201.5858 seconds; validation:27.6004 seconds; recorded waiting:26.7277 seconds; total elapsed:1309.6752 seconds.
- Final fixed-panel metrics: active cosine 0.99237506, waveform MAE 0.0026146482, mel 0.14105883, group MSE 0.0014827562, quiet RMS 78.7338 millionths of full scale, 1242/2544 quiet windows passed, zero overshoot samples.

Compared with the original 2,000-step result, final waveform MAE improved 1.41%, mel 12.32%, group MSE 38.32%, and quiet RMS 10.39%. B did not match the original 5,000-step waveform quality: 94 of 96 recordings had worse MAE. All 13 startup windows still failed; 35 of 171 other-near windows failed only the output-level check, and 1254 of 2360 remaining quiet windows failed. See [findings](findings.md) for the disjoint counts and overlapping cohort comparison.

The separately authorized combined B-plus-startup initializer and 2,500-update recovery had not launched when this completion entry was written. This completed B run remains unchanged. Raw source-level evidence stays on Runpod; the local completion receipt contains aggregate results only.
