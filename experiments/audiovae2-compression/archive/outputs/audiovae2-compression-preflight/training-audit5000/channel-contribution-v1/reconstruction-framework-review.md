# A systematic recovery and pruning procedure

The defensible goal is to reconstruct the teacher's complete function with cheaper retained operations, not preserve every hidden activation or assume a same-shaped output guarantees equivalence. The current intervention identifies which removed computations matter; it does not prove all of them can be recovered from the current retained channels.

## Keep the external contract fixed

For the existing stages 2–4 box, preserve **1024 channels at 200 Hz in, 128 channels at 12 kHz out**, latent conditioning, causal alignment and the original frozen suffix. Train the box jointly against the teacher's full output on the same original latent sequence. Its internal channels may learn a different representation. Selected-coordinate intermediate targets are initialization aids, not the definition of a successful compressed box.

## Choose channels by the function they help reconstruct

Weight magnitude alone ignores correlated and complementary channel contributions. The relevant question is which subset, after refitting its outgoing weights, best reconstructs the following computation on representative training inputs. Channel Pruning combines selection and least-squares response reconstruction; ThiNet likewise selects filters using the next layer's behavior. These support a replacement for plain slicing, although neither validates our audio architecture or promises its speed/quality tradeoff. [Channel Pruning, §§3.1–3.3](https://arxiv.org/html/1707.06168v2), [ThiNet](https://arxiv.org/abs/1707.06342).

For residual units, preserve compatible channel indexing across the skip and branch. Recover the full residual-unit output, accounting for the changed identity input; matching the branch alone is insufficient. When the proposed replacement changes the internal basis or removes whole units, target the enclosing full boundary instead of forcing an arbitrary individual-layer correspondence.

## Reconstruct on the inputs the student will actually receive

If previous compressed operations produce `x_student`, fit the next operation to the original teacher output on the same source, using `x_student` as its input. A target recomputed from the damaged input can preserve the damage, while fitting only teacher inputs ignores accumulated drift. This distinction is the central point of asymmetric reconstruction. [Accelerating Very Deep Convolutional Networks, §III-C](https://arxiv.org/html/1505.06798v2).

Initialize sequentially, then jointly recover the whole box using the fixed full-boundary target plus teacher waveform and spectral objectives through the frozen, differentiable suffix. Local feature improvement must survive that suffix. The recent first-mixer fit and exact first-mixer oracle show why local success is not enough.

## The smallest next probe

1. Refit the existing **stage-4 upsampler** on actual narrow-prefix activations to the full teacher upsampler output. Keep its native stride-2, kernel-4 temporal/polyphase structure, causal crop, output channels and shared bias. Any affine fit must fold into the existing operator.
2. If that gives useful held-out improvement, reconstruct stage 3, regenerate student activations, and **refit stage 4**. A stage-4 solution becomes stale when its input changes. Stage-3's narrower output requires a fixed selected-coordinate target during initialization; the ultimate full stage-4 boundary and waveform remain the acceptance criteria.
3. Recover the complete box jointly. Compare identical initialization, data exposure and recovery budget against the existing sliced-initialization control. Do not overwrite adapted step-5000 weights with teacher-coordinate corrections and call it a continuation.

The stage-4 priority follows the largest controlled ablation effect, not a claim that stage 4 alone caused the trained model's later regression. A failed fixed-shape linear reconstruction does not prove the nonlinear student has insufficient capacity.

## Permit further cuts only after measured recovery

Maintain a small set of candidates across measured CPU latency and quality. Accept a further cut only if it recovers within a fixed training budget and preserves absolute teacher-relative waveform/spectral quality, quiet DC and AC behavior, gain, peaks, startup, and transition/phase-sensitive reconstruction. Keep calibration, development and final held-out sources separate. A correlation average alone cannot approve a cut. These are end-of-recovery checks, not prerequisites for enabling the losses needed to recover.

Our current reduction removes channels while preserving all nine residual units and their dilation/history structure. Removing residual units later is a different change: it can remove temporal context and nonlinear composition. It requires a receptive-field audit and full-box recovery, not merely shape checks. Full-width residual states with low-rank mixing are another candidate if channel selection cannot retain enough information, but extra operator calls and intermediate buffers mean their CPU benefit must be measured.

The current estimated 42.45% MAC reduction is a static budget, not a demonstrated RTF gain. No new model, training or benchmark was run for this review.
