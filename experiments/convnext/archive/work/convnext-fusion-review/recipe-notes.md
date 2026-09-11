# AudioVAE methodology review for the lightweight student

Read-only review, 2026-09-09. No training changes or benchmarks.

## Evidence boundary

- Confirmed AudioVAE architecture: [VoxCPM2 report, section 3.2](https://arxiv.org/html/2606.06928v1#S3.SS2), plus exact decoder source examined by the architecture reviewer.
- Confirmed **older** AudioVAE training disclosure: collaborator Labmem-Zhouyx, [losses](https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3767009845) and [schedule](https://github.com/OpenBMB/VoxCPM/issues/145#issuecomment-3771286634). Read live through GitHub's API because rendered issue omitted comments.
- [V2-specific reply](https://github.com/OpenBMB/VoxCPM/issues/353#issuecomment-4913099637) claims extra <=8 kHz mel, revised KL and training schedule. Author Numanor has GitHub `author_association: NONE` and was not in the current public contributor listing. This is an unverified attribution, not proof the claim is wrong. Do not call it a verified official V2 recipe.
- The released V2 paper does not provide detailed codec loss settings. Its full-TTS training data and RTF cannot be assigned to the VAE.

## Prioritized transfer candidates

### 1. Preserve the teacher output bound with an end-of-decoder tanh

The exact teacher has tanh and the student lacks it, per the architecture review. This is a concrete output contract difference. Tanh is a sensible first candidate to train into the existing student, not a proof of improved fidelity. It does not fix near-zero noise because tanh(x) is approximately x there. Adding it to an already trained unbounded head compresses loud samples initially, so retain teacher waveform losses and fine-tune before accepting it. The training and inference path must both use the same head. It costs one samplewise inference operation and no extra buffering.

Do not add a peak penalty at the same time. Tanh guarantees a range, not preservation of the teacher's transient shape. Validate loud speech and laughter, including pre-tanh logits/saturation, rather than accepting solely because post-tanh peaks fit the range.

### 2. Restore short-time reconstruction coverage within the existing mel objective

Current `reconstruction_v2.py:21-26,105-109` has FFT sizes 1024/2048/4096, hops 256/512/1024 at 48 kHz, plus linear and natural-log magnitude terms. Its shortest analysis window is 21.33 ms, with 5.33 ms hop.

The confirmed older AudioVAE recipe follows DAC's seven scales 32/64/128/256/512/1024/2048, 5/10/20/40/80/160/320 mel bins, hop window/4, log-only magnitude weight zero. DAC's [paper sections 3.5 and 4.5](https://arxiv.org/html/2306.06546v2) explicitly associates low-hop scales with rapid transient and high-frequency reconstruction. This is more directly motivated than a bespoke laughter or quiet loss. Recommend borrowing the short-time coverage concept while retaining existing long scales and valid-frame/sample accounting. The exact resolution set at 48 kHz needs a declared test; older published coefficients cannot simply be copied into different reductions and log bases.

Compute the combined multiscale mel objective within the existing mel branch. Do not make every new scale an independently normalized gradient-balancer branch. All added analysis runs during training, never in the deployed decoder.

### 3. A complex, frequency-band discriminator is a stronger established training-only candidate

Current `discriminators.py:83-108` computes log(abs(STFT)) and has no subband split, 16 channels, versus DAC's real/imaginary two-channel STFT, five independently processed bands, 32 channels. Current MPD widths are also half DAC's widths. Merely calling both MPD+MRD obscures these differences.

DAC's [official discriminator](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/discriminator.py) and paper explain the complex representation as phase-sensitive and subbands as useful for high-frequency aliasing. This is a useful next candidate if periodic or spectral residue remains after the simpler head/mel changes. No inference RTF cost, but meaningfully higher training compute. A fresh discriminator needs warmup; cannot exact-load the current discriminator state or silently call it an exact resume. Keep the learned generator and teacher.

**Do not copy DAC discriminator preprocessing wholesale.** It removes each clip's DC and rescales each clip by its own maximum to a peak of 0.8. That loses absolute amplitude evidence and can amplify near-silence, exactly the signals involved in the current audit. Retain the direct unnormalized waveform reconstruction and deliberately choose amplitude handling.

## Correct current choices to retain

- Frozen 64-channel raw posterior-mean latents and full causal teacher history. `teacher.py:219-237` uses unchanged mean encoding, no implicit amplitude normalization, explicit 48 kHz condition and frozen FP32 teacher. Student latents already match by identity, not by an auxiliary loss. There is no missing KL decoder loss: KL regularizes encoder distribution and contributes no decoder gradient when the encoder is frozen.
- GAN and feature matching already train against the teacher reconstruction at the exact same latent. `recipe_v2.py:162-185` keeps teacher waveform/mel and teacher-as-real GAN/FM active. This is legitimate decoder distillation. Feature matching here means the learned discriminator's features, not the teacher decoder's hidden activations. Arbitrary teacher hidden-channel matching is not required and would require rate/channel adapters.
- Current `distillation_training.py:306-317` chooses an arbitrary audio-sample start for aligned 190 ms adversarial crops, not a multiple-of-480 phase. No fixed-phase discriminator crop bug. A short random crop can miss a transient in a particular update; that alone is not evidence of a broken recipe.
- Reconstruction groups exclude padding/history and weight by actual samples/frames. Do not restore inverse-RMS weighting or blanket loudness normalization. Any gain/phase augmentation must happen before the frozen teacher encoder and generate new matching targets; independently augmenting target audio or scaling latents breaks the pair.
- Teacher FP32, no autocast for STFT, finite checks and identical valid masks are consistent quality safeguards. The DAC base config also sets AMP false; this is not proof the unpublished V2 train used identical precision.
- V2's bandwidth conditioning is fixed at 48 kHz in this student. No need to add trainable multi-SR branches unless the intended inference API needs them.

## Lower-priority differences and caveats

- The older official AudioVAE recipe used 2-second clips, batch 128, 1M updates, 3e-5 cosine LR with 1k warmup, generator clip 1000/discriminator clip 10. Current student uses Muon+AdamW, 2e-4 constant-after-warmup, one shared clip threshold 1, an output-gradient balancer and much shorter training. These are deliberate differences, not diagnosed bugs. Do not transplant the old LR/clip thresholds into this smaller, differently normalized optimizer without update-scale evidence. A million upstream updates is not a justification to wait blindly on a student failure.
- The V2 reply's lowband mel idea is plausible for the asymmetric codec. The official V2 paper independently evaluates lowband and fullband fidelity, which supports reporting both. Prefer short-scale reconstruction first; defer extra <=8kHz weighting unless band-resolved student errors demonstrate the need. Overweighting low frequencies can sacrifice the 48 kHz target.
- DAC's paper and current released implementation disagree on some literal details (paper names HingeGAN; code implements squared real/fake losses, config beta2=.99 vs paper .9). Our +/-1 least-squares labels and averaged heads differ again. Neither loss is inherently invalid, but their numeric coefficients cannot be treated as interchangeable. Do not change GAN labels while also changing the output head or spectral objectives.
- Retain teacher-rendered 48 kHz target for fidelity-to-teacher training. Introducing real audio as a new adversarial target is a new optimization goal and may transfer data noise or conflict with the teacher's reconstructed phase. Native-rate originals can support validation and future refinement but should not silently replace teacher targets.
- No confirmed upstream dedicated silence gate, custom quiet penalty, envelope loss, phase penalty, or peak loss was found. Silence residual can remain a convergence issue or an architecture issue; tanh itself does not solve it. Keep the frozen-teacher encoded-silence, natural quiet, whisper and breath panel when evaluating any candidate.

## Suggested decision sequence

1. Evaluate the already trained current checkpoint; preserve it as control.
2. Introduce only the teacher-style bounded output head, fine-tune with unchanged objectives, and judge fidelity plus range.
3. Add short-time mel coverage if transient/quiet spectral errors remain; keep waveform and longer-band reconstruction.
4. If periodic/phase/high-frequency artifacts remain, replace the magnitude-only MRD with a deliberately adapted DAC complex/subband discriminator.

The useful fusion is the cheap causal ConvNeXt waveform generator plus the teacher's latent/output contract and richer training supervision. It does not require copying the teacher's computationally expensive generator blocks.
