# Why the silence pattern occurs

The current decoder can represent the teacher's stationary silence without adding a layer. Its architecture permits a repeating residual, but the diagnostic does not find a representational barrier for this fixture. The persistent error belongs to the current learned mapping and training behavior.

## What the current checkpoint shows

These results use the preserved step 8,890 checkpoint and corrected canonical teacher targets. No weights or optimizer state were changed.

| Test | Result | Interpretation |
|---|---:|---|
| Explicit layer replay versus ordinary forward | Exact equality | The trace follows the actual decoder |
| Change startup history, compare after the 1.16 s receptive field | Zero difference | Startup handling does not explain the persistent residual in this fixture |
| Replace encoded silence with its stationary latent, compare after settling | Zero difference | The stationary learned response reproduces the residual |
| Existing final-head phase feature matrix | Rank 4, condition number 65.31 | The teacher's four phase targets are representable with the current features |
| External minimum readout correction | 2.17% of current readout weight norm | A mathematical solution exists; its effect on other audio is untested |
| Final projection, FP64 versus FP32 | RMS difference 1.39e-10 | Final-projection rounding is far below the 4.38e-5 residual |

The rank result remains unchanged when singular values below a relative threshold of 1e-4 are discarded. The algebraic residual is about 2.64e-20 in FP64. That number is a feasibility calculation, not measured audio after a trained or deployed correction. No fitted readout was installed.

## The architectural mechanism

The last head maps each 100 Hz hidden frame to 480 independent sample positions:

`audio[480*n + r] = output_weight[r] dot hidden[n]`

For a constant hidden vector, each output position can still receive a different value. Repeating that frame produces a 10 ms pattern, with spectral energy on a grid spaced by 100 Hz. The four internal phases per 40 ms encoder latent allow additional 40 ms structure.

The actual steady encoded-silence latent is nonzero, with RMS 0.6474. Its four adapted phases differ, and those differences pass through the temporal network. Every traced stage is stationary across complete cycles. The later blocks strongly suppress the phase differences, but the final head still maps its nearly stationary features to a non-flat waveform.

For the student-minus-teacher steady residual, 93.19% of power belongs to the shared 480-sample pattern, 5.48% is DC and 1.34% is additional 1,920-sample structure. The teacher's steady silence is itself nonzero, with RMS 9.63e-6 and almost entirely DC. Correct distillation therefore targets this learned teacher response, rather than assuming zero audio means zero latents or requiring an arbitrary latent vector to decode to zero.

This mechanism is described directly in Pons et al.'s work on neural audio upsampling: convolution followed by a periodic channel shuffle can introduce tones because neighboring output positions use different filters. It does not require overlapping transposed convolutions. Their analysis also explains why special initialization alone cannot prevent artifacts returning during training, and why interpolation replacements can alter frequency response. [Paper](https://arxiv.org/abs/2010.14356), [authors' runnable explanation](https://github.com/DolbyLaboratories/neural-upsampling-artifacts-audio/blob/main/ARTICLE.md#4-subpixel-convolutions)

## Natural quiet audio remains a different problem

The earlier matched decomposition found that 81.93% of natural-quiet residual power varies with the input. A universal stationary-silence subtraction improved natural-quiet RMS by only 0.65%.

The new six-source check selects recordings by teacher-defined quiet coverage before inspecting student errors. Replacing each recording's varying latent trajectory with one repeated latent makes its error against the original teacher larger in all six cases. Those counterfactuals no longer encode the original audio, so they are not valid reconstruction candidates. They demonstrate why preserving low-level temporal information matters; treating quiet audio as a constant silence response discards that information. This selected set is concentrated in Mandarin, Cantonese and German, plus one expressive recording; it is not a representative multilingual quality evaluation.

## What the reference architectures suggest

Supertonic's published decoder uses a direct time-domain head and flattening, the same broad head family. This is an efficient synthesis design, not a guarantee that it reconstructs every frozen AudioVAE2 latent without periodic errors. [Published decoder](https://arxiv.org/html/2503.23108v3#A1.SS1.SSS2)

AudioVAE2 uses progressive causal upsampling, Snake activations and a final waveform convolution followed by tanh. Its bounded output helps with full-scale limits. Tanh is approximately identity near zero, so it does not inherently remove a quiet residual. Neither learned biases nor weight normalization guarantees exact silence. [Official AudioVAE2 source](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/modules/audiovae/audio_vae_v2.py)

The newer WaveNeXt 2 adds iterative residual refinement. Its published centered-STFT conditioning and extra submodels are a substantial change to the streaming and compute contract, rather than a demonstrated fix for stationary silence. Its own table reports higher RTF than base WaveNeXt. No AudioVAE2 latent or digital-silence result establishes a benefit here. [WaveNeXt 2](https://arxiv.org/html/2605.25506v1#S3)

## Decision

Keep the architecture fixed for the optimizer comparison. The present evidence supports investigating how updates move the learned silence response, rather than installing another layer. A head-only solution to one fixture does not establish acceptable reconstruction elsewhere, and a strict output amplitude guarantee remains a separate design requirement.

[Machine-readable evidence](architecture.json) · [Predeclared experiment protocol](protocol.md)
