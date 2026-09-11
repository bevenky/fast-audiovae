**Teacher preparation issue found during recovery**

The two inconsistent laughter encodings came from one historical eight-recording batch. Replaying that exact batch reproduces all eight archived latent tensors bit for bit. Running the same audio individually gives different values in the last latent channel. This is a reproducible backend execution defect on the tested Runpod environment, not a changed recording or checkpoint.

The tested environment is H100, PyTorch 2.11.0+cu128 and cuDNN 9.19. Both TF32 flags were disabled and deterministic algorithms were enabled. These settings alone did not prevent the failure. This observation does not establish that every system with those versions fails.

| Check | Result |
| --- | --- |
| Historical batch | Eight recordings, each padded to 93,440 input samples. All eight cached latent prefixes reproduce exactly. |
| Same padded recording executed individually | Material differences from the historical batch; removing the extra padding changes results only at numerical precision. |
| First divergent encoder layer | `encoder.fc_mu`, the final posterior-mean projection. Its input and effective weights are identical in the two executions. |
| Affected projection output | Channel 63, the last of 64. On the traced laughter source, maximum difference is 1.19375. The other channels differ only at approximately FP32 rounding scale. |
| Log-variance projection | Also differs in channel 63, by up to 7.13476. This output is not used for the posterior-mean student targets. |
| High-precision arithmetic reference | Historical batched `fc_mu` differs by up to 1.19375; the individual result differs by at most 1.43e-6. |
| cuDNN bypassed from startup | Batched and individual encoding agree to less than 7e-7 RMS on all eight recordings. |
| TorchScript optimization disabled from startup | Did not remove the discrepancy. |

The result depends on execution history: in a separate process where singleton shapes ran first, the subsequent eight-recording batch matched the singleton result closely. Merely selecting deterministic execution or warming up an unrelated shape is therefore insufficient. Disabling PyTorch's cuDNN execution-plan cache did not fix it. We isolated the failing path, but have not identified the exact internal cuDNN implementation defect.

**Upgrade result:** changing only cuDNN to 9.25.1, while preserving PyTorch 2.11.0 and the original tensors, fixed the isolated projection. Its maximum error against the CPU FP64 reference fell from 1.19375 to 0.0000117, within the pre-existing tolerance. The complete eight-source encoder reproduction then produced bitwise-identical batched and singleton outputs. This establishes a working library upgrade for the demonstrated failure, rather than requiring a permanent cuDNN bypass.

The earlier batch qualification used five different recordings padded to a different length. That passing result was reused to authorize subsequent batches without measuring each new batch against individual execution. Neither laughter source was in the qualification panel. This was a gap in our preparation checks.

**Recovery scope**

The next environment must use the verified corrected backend and record its versions and execution policy in cache provenance. Each actual batch must also pass numerical checks before publication. A cuDNN bypass remains a tested fallback. Existing checkpoints and cached pairs are preserved.

Checks on eight cached windows support latent-to-teacher-waveform consistency for those windows; they do not certify every historical target. The encoder defect establishes a mismatch with the intended input distribution. A separately versioned panel was reconstructed from all 147 authentic held-out sources. Exactly eight sources had latent differences above 0.001; these were the known failing batch. The teacher-defined quiet window count remained 4,073. Historical metrics and caches were not overwritten.

The bounded student/discriminator audit did not reproduce this shape-dependent defect. Within each backend, batch-size and repeated-shape outputs matched bitwise. Across cuDNN-on and off execution, the whole balanced student parameter gradient differed by 0.0166%, and the discriminator parameter gradient by 0.000592%. All original engine, optimizer, EMA and checkpoint states were preserved. This does not certify every historical training shape.

The original 10,000-step run used singleton teacher preparation. The current fresh model lineage used batch-capable preparation for 166.50 scored hours through step 8,090; the exact fraction affected cannot be recovered from its incomplete per-batch provenance. Preserve these checkpoints and evaluate on corrected encoder inputs before deciding continuation or restart. This finding does not justify labeling every earlier checkpoint invalid.

Evidence: [exact historical batch](historical-batch.json), [layer trace summary](layer-trace-summary.json), [FP64 projection reference](projection-reference-summary.json), [backend comparisons](backend.json), [TorchScript-disabled startup](jit-from-start.json), and [cuDNN-disabled startup](cudnn-from-start.json). All retained checkpoint hashes remained unchanged during these checks.
