# What channel pruning changed, and what should happen next

The next decision should address the pruning method itself before a longer continuation. The larger-accumulation comparison demonstrated a useful training-policy effect, but it did not establish that the chosen channels preserve the teacher's important computations. The earlier audits localized reconstruction errors to stages 2–4 and ruled out several wiring defects. They did not identify the particular removed contributions responsible for each failure.

This review reads the original papers, released decoder, compression implementation and saved measurements. It runs no new models, training, tests or benchmarks. The main run remains paused.

## Original decoder and the exact change

AudioVAE2 encodes 16 kHz audio into 64-channel latent frames at 25 Hz. Its released decoder expands those through six causal upsampling stages with strides 8, 6, 5, 2, 2 and 2, ending at 48 kHz. Each stage contains an input Snake, a transposed convolution and three residual units. Each residual unit contains channelwise Snake, a causal depthwise filter, another Snake, pointwise channel mixing and a skip connection. The final neighboring-sample convolution and tanh remain in the student. The V2 report explicitly describes its wider decoder as supporting higher reconstruction bandwidth. [V2 report, section 3.2](https://arxiv.org/html/2606.06928v1#S3.SS2), [released implementation](https://raw.githubusercontent.com/OpenBMB/VoxCPM/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py).

The current student preserves all nine residual units in stages 2–4 and all original temporal strides, dilations, padding and causal support. Its internal channels were reduced simultaneously at two boundaries:

| Operation | Teacher | Student | What is removed |
|---|---|---|---|
| Stage 2 upsampler | 1,024 inputs → 512 outputs, 12 taps | 1,024 → 256, 12 taps | Half the output filters |
| Each stage 2 pointwise matrix | 512 × 512 | 256 × 256 | 75% of scalar mixing connections |
| Stage 3 upsampler | 512 → 256, 10 taps | 256 → 128, 10 taps | Half the inputs and outputs; 75% of scalar weights |
| Each stage 3 pointwise matrix | 256 × 256 | 128 × 128 | 75% of scalar mixing connections |
| Stage 4 upsampler | 256 → 128, 4 taps | 128 → 128, 4 taps | Half the input-channel contributions |
| Stage 4 residual stack | 128 channels | 128 channels | No width reduction; it now receives changed inputs |

The associated depthwise filters, per-channel Snake parameters and relevant conditioning coordinates are also selected consistently. The encoder, prefix and suffix are frozen. Original teacher training co-adapted a different, wider function; our recovery is constrained to the smaller middle group. This is deliberate, but it concentrates the burden of compensation inside that group.

## The first missing computation is identifiable

At fresh sliced initialization, the selected stage-2 upsampler outputs reproduce the corresponding teacher coordinates, subject to numerical precision. Its first residual unit's channelwise operations preserve that correspondence up to the pointwise mixer. Partition the mixer inputs into kept K and dropped D:

`teacher kept output = kept skip + W_KK × kept features + W_KD × dropped features + bias_K`

`student output = kept skip + W_KK × kept features + bias_K`

The term `W_KD × dropped features` is omitted. The initializer does not approximate it using kept features and does not fold its predictable mean into the bias. This establishes a concrete change in the function. It does not yet establish which omitted terms cause the trained student's silence or whistle error.

After this first mixer, later Snake activations receive different inputs. Removing channels therefore changes subsequent nonlinear transformations, not just the final linear sum. At the next two upsamplers, removed input channels also lose their learned contributions to different output phases. The phase routing and causal trim themselves remain correct; whether the lost contributions produce a particular spectral artifact is unmeasured.

## The channel-selection limitation

The code uses 72 calibration recordings and at most 256 randomly selected scored time cells per recording, separately at the original stage-2 and stage-3 outputs. A pivoted uncentered activation Gram selects coordinates. The same coordinates are then retained throughout each stage's three residual units.

This measures activation coverage at the end of a stage. It does not measure the effects of discarded coordinates on the next layer or final waveform. It also does not establish redundancy inside every residual unit, protect specific upsampling phases or event intervals, or fit a replacement for discarded outgoing contributions. Different channels contain different learned temporal filters and Snake functions; similarity at the stage output does not establish that those internal functions are interchangeable.

The calibration set includes speech and expressive material. Its size or composition alone is not proven insufficient. The specific gap is that selection has no downstream reconstruction criterion or recorded bound on the omitted contribution.

## Which side effects fit the observations?

| Behavior | Architectural explanation worth investigating | What the evidence actually establishes |
|---|---|---|
| Stationary near-silence mismatch | Removed weighted contributions could alter the tiny final mean or a cancellation | Teacher hidden features are nonzero during near-silence, and the student has small mean/residual deviations. Signed weighted contributions and cancellation have not been measured. Large hidden values alone do not prove cancellation. |
| Startup and transients | Removing learned causal filter responses can change the response to the same initial history | A larger startup error and a missed teacher transient were measured. Padding/history are retained; the responsible filter or removed channel is unknown. |
| Whistling and expressive detail | Fewer nonlinear/filter components or different upsampling-phase sums may be harder to approximate | Whistle level changes from too loud to too quiet across checkpoints. No permanent bandwidth, phase or amplitude ceiling is established. |
| Broad gain swings | Group-output changes pass through a sensitive fixed suffix | Same-architecture accumulation twelve reduces the swings. That is evidence of a training-policy contribution, not proof that compression is harmless. |

Some near-silence failures are tiny amplitude-threshold violations, not established audible noise. On one recorded intervention, reducing the stage-4 error lowers final residual RMS but increases near-silence failure counts. Those counts alone cannot demonstrate architectural deterioration. Preserve the checks, but distinguish residual, signed mean, amplitude ceiling, onset error and perceptual consequence.

## Will training alone repair it?

There is evidence of ongoing recovery: from step 1,000 to 5,000, group MSE falls about 53% and mel error about 32%. The later accumulation comparison improves waveform fidelity with the same architecture. This argues against declaring the student incapable of learning.

There is no evidence that all failures will resolve just by adding steps. Near-silence pass counts deteriorated early, around step 1,500, and remain poor. Latest larger-accumulation training barely changes quiet residual RMS or the whistle's average gain error. These are not exclusively a new late-run phenomenon. Conversely, we do not have a converged same-width fit or controlled capacity comparison proving that they are irreducible architectural limits.

Halving channels does not prove that half the information is irretrievably lost: temporal upsampling increases the number of feature values, and inputs come from the structured latent sequence. The definite loss is the removal of specific computations from the copied teacher. Whether the remaining network can re-express them requires evidence.

## Recommendation: explain the removed contributions before more long training

1. **Measure the omitted terms directly.** Use identical teacher inputs and causal history at the first stage-2 mixing operation and at the stage-3/4 upsamplers. Split retained and dropped contributions, including signs, means and phase-specific output. Cover stationary quiet, onset, ordinary speech and the actual whistle/transient intervals.
2. **Use controlled restorations to establish an effect.** Restore one omitted contribution in a clearly defined teacher/pruned-initialization diagnostic and propagate it through the otherwise fixed downstream path. First establish that the masked diagnostic reproduces the corresponding sliced computation. Observe whether the actual waveform failure changes, rather than only the feature MSE. Such restorations explain local effects; they are not deployable repairs or capacity proofs.
3. **Keep initialization and trained representations distinct.** At step 5,000, the narrower channels may have changed meaning. Pasting selected teacher terms directly into them would not be a valid causal attribution. Use the shared complete stage-4 boundary for that model and separately assess which initial pruning effects remain after learning.
4. **Choose a compression change from that result.** If omitted contributions are reconstructible from retained features, compare reconstruction-aware fitting of the same matrices against plain slicing. This adds no inference operation. If important contributions remain unrecoverable under a controlled fit, test one selectively restored width or a student that retains full-width nonlinear residual states and factors the expensive linear matrices. Rank and CPU benefit must be measured; neither design is a guaranteed cure.

Keep the existing checkpoints and the demonstrated accumulation improvement. The recommendation is to pause the proposed long continuation while closing the specific channel-attribution gap. No new architecture should be selected merely from the current pass counts or from the fact that all module types remain present.

Supporting audits: [operation and tensor audit](architecture-causal-audit.md), [tensor and channel design review](teacher-compression-design-audit.md), [measured evidence and its limits](architecture-evidence-review.md), [saved training history](validation-audit.md), [current accumulation comparison](accumulation-comparison-v1/independent-analysis.md), [conditional student options](conditional-student-options.md).
