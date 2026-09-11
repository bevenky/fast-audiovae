# Distillation guidance for the compressed AudioVAE2 decoder

This records the recommendation after the channel-contribution experiments and the user's request to review student-teacher literature. The approved recovery comparison has now completed: [result and decision](../projected-hints-v1/summary.md). It improved intermediate feature matching without a final reconstruction benefit, so the hints were not adopted. The preserved trained checkpoints remain available; the failed isolated first-mixer patch is also not adopted.

## Relevant precedents

[StreamCodec2](https://arxiv.org/html/2509.13670v1#S2.SS3) explicitly combines a smaller causal codec with projected intermediate teacher guidance. Its up/downsampling-hint ablation is directly relevant to the operations investigated here. It improves its student but does not demonstrate teacher parity or fixed AudioVAE2 latent compatibility.

[FitNets](https://arxiv.org/html/1412.6550v4#S2.SS2) explains using a learned regressor to make a thin student's representation predictive of a wider teacher representation. It also cautions that hint placement can overconstrain the student. [DLL-APNet](https://arxiv.org/html/2509.13667v1#S3.SS3) provides a neural-vocoder precedent for intermediate teacher guidance alongside synthesis objectives. These support testing auxiliary feature supervision, rather than requiring every student channel to equal a selected teacher channel.

[Channel Pruning](https://arxiv.org/html/1707.06168v2#S3.SS2) and [asymmetric reconstruction](https://arxiv.org/html/1505.06798v2#S3.SS3) address a complementary problem: initializing the remaining operations after a cut using the current smaller network's inputs and the original target responses. Initialization and ongoing neural distillation are separate mechanisms.

## What our implementation already does

The current pruned model already learns from the frozen teacher with waveform L1, mel and unprojected MSE at the complete 128-channel stage-4 boundary. Gradients pass through the frozen suffix to the trainable stages 2–4. The encoder and 64-channel latent interface remain fixed. This is already knowledge distillation.

The intermediate stage traces currently provide diagnostics, not losses. The concrete extension to test is a pair of soft training-only hints:

| Location | Student feature | Teacher feature | Proposed loss-only alignment |
|---|---|---|---|
| Stage-3 upsampler output | 128 channels at 6 kHz | 256 channels at 6 kHz | Per-frame linear 128→256 |
| Stage-4 upsampler output, before residual stack | 128 channels at 12 kHz | 128 channels at 12 kHz | Per-frame linear 128→128 |

Both feature pairs have the same temporal grid. Alignment must not mix future frames. The projections exist only on auxiliary training branches and are removed from inference/export. They are not inserted between decoder stages. The deployed decoder's layer count, state and operation shapes therefore stay unchanged.

The complete stage-4 END boundary must still match directly, without a learned projector: the frozen suffix consumes those original coordinates. A low projected hint loss can hide scale or basis differences, so it cannot substitute for direct boundary and waveform checks.

## Controlled next comparison

For the current trained student, prioritize a same-checkpoint recovery comparison of the existing recipe against the existing recipe plus these two hints. This is more directly relevant to remaining trained-model errors than another repair of an untrained initialization. Use identical source order, exposure, optimizer policy and audio objectives. Do not combine this comparison with new pruning, new output filters or normalization changes.

Before the comparison, initialize/calibrate only the auxiliary alignment maps on training data and confirm that their losses backpropagate into the intended student layers. Use valid sample-count weighting, including context exclusion and partial tails. The two feature cells cover eight and four waveform samples respectively. Do not normalize each quiet example by its near-zero audio RMS.

Treat the hints as auxiliary. Set their weighting using training-only gradient measurements with the current reductions, rather than copying a coefficient from a different paper. Track the contribution to student gradients separately from projection gradients. Retain the existing full-boundary, waveform and mel objectives throughout recovery. Near-final correlation is an acceptance target, not a prerequisite for turning these objectives on.

Score raw waveform, active amplitude/correlation, quiet residual and output levels, onsets/transients and expressive audio on the same held-out panel. A lower hint loss is insufficient. Confirm that export contains no auxiliary projections and preserves the existing CPU streaming graph before any adoption. A larger training graph alone does not establish faster convergence or a quality gain.

## Mechanism for subsequent cuts

Once recovery is acceptable, choose the next cut by output sensitivity and measured CPU benefit. Initialize the affected remaining operators through reconstruction on the smaller network's actual inputs, refresh downstream fits after upstream changes, then run joint whole-box distillation with the same final teacher reference. Treat a shorter chain as a replacement for the complete original chain. Do not impose one-to-one internal matches where layers have been removed or coordinates have changed meaning.

For a cut inside an already adapted student, use the preserved pre-cut block function for local coordinate-consistent initialization, or expand the fitted region to an externally aligned boundary. Keep the original AudioVAE2 complete-group output and waveform as the final quality anchor so successive accepted students do not silently become the sole reference.

The earlier proposed native stage-4 reconstruction probe remains useful for investigating the initializer at current widths. It is not the first recovery comparison recommended here after the additional literature review. If hint-based recovery is insufficient, that probe can distinguish predictable lost contributions from upstream feature damage before selecting a different width or matrix factorization.

No distillation method guarantees recovery after arbitrary removal of nonlinear units, history or information. If the agreed quality cannot be recovered within a bounded comparison, reduce the cut or change how the expensive computation is represented. Do not add further pruning merely because one local feature score improves.

Supporting records: [completed experiment](findings-and-compression-plan.md), [code-to-StreamCodec2 comparison](../streamcodec2-method-gap.md), [other audio distillation precedents](codec-distillation-precedents.md), [independent quantitative audit](independent-analysis.md).
