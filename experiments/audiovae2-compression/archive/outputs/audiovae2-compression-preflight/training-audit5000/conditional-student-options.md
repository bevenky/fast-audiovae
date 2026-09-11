# Conditional student options after the update audit

This memo proposes no immediate architecture experiment. First finish the authorized accumulation/update comparison. If it explains the late gain drift, keep the present architecture while correcting that mechanism. A better initializer cannot retrospectively explain a sudden change between updates 4,500 and 5,000.

If a controlled continuation still cannot jointly recover waveform level, quiet audio and the complete group output, the first alternative should preserve the current dimensions and improve how the teacher is compressed. Additional residual-unit deletion is the least justified next change.

## What the evidence actually supports

The current initializer selects coordinates from uncentered activation Gram matrices, then slices incoming and outgoing effective weights. It does not reconstruct removed contributions. In a pointwise convolution, discarded channels can contribute through outgoing weights even when their activations are predictable from retained channels. The code correctly keeps dependencies aligned; that is different from preserving the trained function. See [`run_pilot.py`, lines 207–248](../../../work/fast-audiovae/experiments/audiovae2-compression/run_pilot.py#L207) and [the pruning audit](pruning-reaudit.md).

The intact suffix exactly reproduces the teacher when given true teacher stage-4 features. At 5,000 updates, halving the student's group-error vector reduced active waveform error and excess gain in both targeted languages. Thus the shared boundary is appropriate, but scalar group MSE overlooks error direction. The typical near-silence failure also involves micro-amplitude drift around a nonzero teacher floor, rather than every failure representing loud added noise. These findings do not prove insufficient width or identify a disposable residual unit. [Measured diagnosis](layer-diagnosis.md).

## First conditional comparison: reconstruct the retained weights

Build a separate teacher-initialized 256/128 student, using the same channel indices initially to isolate reconstruction from selection. Before joint fitting, estimate the removed contributions and refit existing outgoing affine weights and biases on training-only calibration examples. At a pointwise layer this is an affine least-squares fit from the current retained input features to the original teacher's selected output. Reconstruct successive operations using the actual compressed upstream activations, rather than assuming those still equal the teacher. Recreate weight-normalization parameters from fitted effective weights.

These local fits are an initialization procedure only. The subsequent learning objective remains the entire stages-2–4 boundary plus final teacher audio; it does not impose ongoing one-to-one hidden-layer targets. There is no added deployed layer or silence gate. Calibration must include quiet and active speech, transients and expressive sources without dividing each quiet example by its tiny RMS. The teacher's measured waveform, including its floor, remains the target.

Channel selection followed by least-squares reconstruction has direct precedent in [Channel Pruning](https://arxiv.org/abs/1707.06168). [ThiNet](https://arxiv.org/abs/1707.06342) instead selects using the next layer's reconstruction. [Asymmetric reconstruction](https://arxiv.org/abs/1505.06798) addresses the use of already-approximated upstream activations. Their CNN results support this initialization question; they do not establish audio quality or solve nonlinear Snake groups exactly.

**Do not apply this initializer over step 4,500.** It would overwrite learned group weights and invalidate a continuation comparison. Preserve that model. Compare fresh teacher-derived plain slicing against fresh teacher-derived reconstruction, with equal training sources, exposure, optimizer initialization and objective. Charge calibration work separately. Change channel selection itself only if reconstruction with the existing indices is demonstrably inadequate.

## If capacity must change

All rows below preserve the outer 1,024-at-200-Hz to 128-at-12-kHz group contract and the original suffix. Counts use the existing static formula, excluding activation, dispatch, memory and streaming overhead.

| Candidate | Decoder GMAC/audio-second | Reduction from teacher | Decision value |
|---|---:|---:|---|
| Current 256/128, nine units | 5.174 | 42.45% | Reconstruction-aware initialization changes no runtime graph |
| Restore stage-3 width 128→160 | 5.541 | 38.38% | Small increase in the second internal boundary's capacity |
| Restore stage-2 width 256→320 | 5.564 | 38.12% | Similar arithmetic budget at the first boundary |
| Full-width group with illustrative low-rank projections | 6.285 | 30.10% | Preserves channel coordinates and full-width nonlinear paths |
| Current widths, six residual units | 4.782 | 46.81% | Only another 7.57% below current MACs; reduces temporal support |

If restoring width becomes necessary, select **one** boundary using training-only downstream reconstruction evidence and fixed held-out confirmation, not the largest internal selected-coordinate error. Retain all nine units and all dilations. These counts do not establish an Intel RTF gain; even a 38% MAC reduction must pass actual one-thread streaming execution and quality checks.

The low-rank alternative keeps full-width Snake, depthwise convolutions and residual state. A pointwise matrix becomes `C→r→C` with no intervening activation. A transposed convolution can factor its input-time matrix `Ci→(2sCo)` into `Ci→r→(2sCo)`, retaining exact phase routing, overlap and trim. It is not interchangeable with two arbitrary waveform-rate 1×1 convolutions. The illustrative count uses `r=Ci/2` for group upsamplers and `r=Co/4` for group pointwise matrices. These ranks have not been selected or tested. Training-only activation-weighted reconstruction and singular spectra must show that the important downstream directions survive. Extra calls/buffers and unchanged full-width Snake work may offset the arithmetic savings.

Keep whichever conditional candidate improves final waveform and regional errors together under matched exposure. Reject a candidate that only improves feature MSE or global correlation while amplitude, startup or expressive behavior regresses. An unsuccessful short recovery shows that a particular proposal failed its budget, not that all smaller AudioVAE2 decoders are impossible. No architecture test, model update or benchmark was run for this memo.
