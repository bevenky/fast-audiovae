# GRAIL-like hidden-feature compensation

Authorized independent2,000-update trial.34 local and34 qualified Runpod CPU tests passed, with independent method/runner review. The finite queue will start it after the downstream-selection trial completes.

- Source archive SHA256 `31b063580135ecde46139fb7c1616762e6dd434f20628125cb5aa0899a5bca8e`,33,095bytes.
- Qualified CPU test log SHA256 `4a444e19c7490e628229640a741fe91b7bf4da170c64ccf19211a3b027cfd936`.
- Source/initializer root `/workspace/fast-audiovae-grail-hidden-20260911-v1`.
- Recovery root `/dev/shm/fast-audiovae-grail-recovery-20260911-v1`. Runpod shared memory holds the new optimizer checkpoints because the existing disk volumes lack sufficient free space. Keep the pod running to retain these files. No prior data or artifacts were deleted.
- Fresh original teacher factory with the original pivoted384/256 selection; no trained checkpoint weights. Fit four sequential post-Snake hidden maps on the original72 calibration sources, fold into the existing native consumers and preserve native bias.
- The same map is shared across all ten upsampler taps. No affine feature offset, added module, startup constraint or new training loss.
- Original first24,000 distinct sources, physical batch1/accumulation12, all90group tensors, fresh original Adam/RNG, exactly2,000 updates. Fixed96development reviews0/250/500/1000/1500/2000.
- Comparator is B at matched source exposure. Maps are retained as diagnostic artifacts; inference uses only the native folded operators.

The code-only upload verification initially hit a path-expression typo in the shell wrapper, before any tests or model execution. The wrapper was corrected, all original source hashes matched, and34 remote tests passed. Model source was unchanged.

[Fixed method and limits](../follow-on-experiments-plan.md). No GPU result is claimed in this launch note.

Operational update: follow the [active corrected queue](../independent-trials-v2/execution.md). GRAIL initialization output moved to `/dev/shm/fast-audiovae-grail-hidden-20260911-v1/results`; source code and method are unchanged.

Completion update, 11 September 2026: G completed exactly 2,000 updates and saved its checkpoint. The aggregate CPU audit passed. Waveform MAE improved 5.29% versus B, quiet passes improved to 1,419/2,544, and startup RMS fell 65.77%, but startup remains 0/13 and the 20–40 ms teacher transient worsened. See [completed findings](completed-findings.md) and [aggregate evidence](completed-aggregate.json). The quiet-preservation arm has launched independently; G is preserved without promotion or extension.
