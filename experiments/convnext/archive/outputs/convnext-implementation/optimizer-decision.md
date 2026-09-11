# Optimizer choice for the AudioVAE2 student

Decision, 8 September 2026: the completed 1,000-update speech warmup uses the
Muon/AdamW hybrid. The authorized continuation retains both optimizer states
and the learning rate, increases the true batch to 64, and uses fresh speech
through step 10,000. A matched AdamW comparison would be needed to claim an
optimizer convergence advantage; it is deferred while this run continues.
AudioVAE2 remains the only teacher.

## Different roles

The teacher produces target audio from the frozen encoder's latents. Losses
measure student errors; backpropagation computes gradients; the optimizer uses
those gradients to update the student. The teacher does not replace an optimizer.

Freezing the encoder avoids learning a moving latent representation. Teacher
outputs provide a fixed mapping, including high-frequency synthesis the 16 kHz
input alone does not specify. This can simplify training, but does not guarantee
a particular convergence speed or data reduction. Online teacher decoding also
costs compute; cached targets trade that compute for storage and I/O.

## What the AudioVAE literature establishes

The earlier AudioVAE author discussion gives mel, feature matching, adversarial
and KL losses; a separate collaborator reply gives a cosine schedule with
1000-step warmup. These are earlier-version details, not a complete published
AudioVAE2 optimizer recipe. [Loss discussion](https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3767009845),
[schedule discussion](https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3771286634).

The VoxCPM2 paper reports AdamW for TTS backbone training. That should not be
presented as evidence of the optimizer used to train AudioVAE2. The public
fine-tuning path freezes the VAE. A V2-specific issue reply provides additional
recipe claims, but the reply's author was not verified as a project author;
those claims do not define this experiment's defaults.
[VoxCPM2 paper](https://arxiv.org/html/2606.06928v1#S3.SS5),
[V2 discussion](https://github.com/OpenBMB/VoxCPM/issues/353#issuecomment-4913099637).

Our frozen encoder needs no new KL objective or encoder optimizer.

## Why test Muon

Qwen-Music's Spec-VAE uses Muon for reconstruction pretraining and the subsequent
adversarial stage. Its later AdamW stage trains a separate refiner with the VAE
frozen. This is useful audio precedent, but it is not a controlled speedup study
for our causal waveform decoder or evidence for switching the same decoder from
Muon to AdamW halfway through training.
[Section 2.4.2, Table 2](https://arxiv.org/html/2607.11699v2#S2.SS4.SSS2).

The hybrid runs both optimizers on disjoint parameters at every update:

| Parameters | Optimizer |
|---|---|
| Twenty `blocks.*.expand.weight` / `blocks.*.project.weight` matrices | Muon |
| Input adapter, stem, waveform head and depthwise filters | AdamW |
| Biases, normalization, affine gains, residual scales and PReLU | AdamW |
| Discriminators, once implemented | AdamW in both comparison arms |
| Frozen encoder and teacher decoder | None |

The selected hidden matrices are already 2D and contain 20,971,520 weights.
PyTorch's native Muon accepts 2D parameters. The original reference implementation
reshapes 4D convolution weights but does not provide the required mapping for
our 3D Conv1d filters. Do not select every parameter with `ndim >= 2` or flatten
depthwise filters indiscriminately.
[Native API](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html),
[reference implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py).

Current native Muon settings: `adjust_lr_fn="match_rms_adamw"`, momentum 0.95,
Nesterov, five Newton-Schulz steps and an explicitly shared starting learning-rate
schedule. Do not combine the original implementation's 0.02 learning rate with
RMS-matched scaling. Learning-rate suitability for this model remains empirical.
Muon orthogonalizes updates, not the trained weight matrices themselves.

## Comparison and adoption

Start both arms from the same saved fresh initialization. Hold data order,
teacher targets, source balance, crop/context alignment, effective batch, losses,
precision and hardware fixed. Compare reconstruction training first, then a
matched early adversarial window before selecting the sustained-pilot optimizer.
Each parameter must belong to exactly one optimizer, and both optimizer states
must be saved and restored with the same accumulated-gradient update counter.

Select by held-out audio quality per elapsed H100 hour, including step overhead,
data preparation and validation. Also plot progress per update and record
language/source breakdowns. A lower training loss alone is insufficient.

Muon has extra matrix operations per update. Fewer updates only help wall time
if they compensate for that overhead. Published language-model efficiency gains
cannot be assigned to this decoder without measurement. The optimizer is absent
from inference, so choosing Muon adds no CPU decoding operations or direct RTF
benefit.

The hybrid is implemented and its parameter partition, both optimizer states,
CPU checkpoint replay and TensorBoard event history have passed checks on
Runpod with PyTorch 2.11.0+cu128. This is the newest official CUDA 12.8 build
found for the existing driver. The local CPU validation environment retains
PyTorch 2.8, where native-Muon execution checks are explicitly skipped.

Both optimizers completed 32 synthetic updates of the full-capacity student on
the H100. Those runs validate training and live loss logging, not speech
convergence or a Muon advantage. Original AudioVAE2 target parity on H100 now
passes, and all 255 complete-utterance targets are cached. The hybrid completed
1,000 speech updates using 230 training clips across 12 languages. Fixed-crop
validation on 25 speaker-disjoint LibriSpeech clips fell from 171.89 to 40.89.
This is reconstruction loss, not perceptual quality or a matched optimizer
comparison. The latter remains outstanding. Training and held-out validation
have separate TensorBoard curves under `warmup-speech-muon-v1`.
