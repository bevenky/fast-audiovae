# Result: the decoder passed the 0.99 reconstruction diagnostic

**Follow-up quiet-audio audit:** The declared whole-crop diagnostic passed, but stricter 20 ms checks expose remaining quiet-region errors even within fitted clips. The [quiet-audio findings and prepared corrections](quiet-audio-findings.md) are part of the remaining quality work. The checkpoint is not a final quality pass.

The selected target remains **0.99 per nonquiet fitted crop**, with amplitude, waveform-error and quiet-audio checks. The waveform-only arm passed the complete diagnostic at two consecutive evaluations, after 1,800 and 1,900 additional updates. Both bounded runs have finished. The representative-data pilot and perceptual stage have not started.

| Fitted diagnostic result | Waveform plus mel | Waveform only |
| --- | ---: | ---: |
| Additional updates from the same step-500 parent | 2,000 | 1,900 |
| Mean nonquiet waveform cosine | 0.99645 | **0.99856** |
| Lowest nonquiet waveform cosine | 0.95609 | **0.99401** |
| Nonquiet crops with cosine at least 0.99 | 29 / 31 | **31 / 31** |
| Nonquiet crops within 1 dB of teacher RMS | 30 / 31 | **31 / 31** |
| Quiet fitted crop | Passed | Passed |
| Complete reconstruction acceptance | Failed | **Passed twice** |
| Mean teacher mel error over all fitted crops, lower is better | **0.54534** | 0.67724 |
| H100 run time | 307.8 seconds | 278.8 seconds |

These measurements cover 16 fitted utterances at two positions each. They establish that this full lightweight decoder can learn the original decoder's mapping from the original raw latents. They do not establish multilingual generalization or teacher-quality perceptual audio.

## What the comparison establishes

Both runs started from identical model weights, optimizer moments and counters, frozen normalization, moving-average state and random-number states. They received exactly the same cached 64-channel latent tensors and teacher waveform targets, in the same order. The teacher and encoder were not updated or rerun. Independent verification confirmed the original checkpoint, cache metadata, index and all 32 cached utterance files remained unchanged.

Both arms used a 50-update learning-rate ramp from 0.00002 to 0.0002, then a constant rate. They differed in active waveform/mel gradient shares: 0.5/0.5 versus 1.0/0.0. Parameter-gradient reports used the actual full batch, and sampled probes recorded actual optimizer weight changes. The forward model was unchanged.

Waveform-only reached the fitted acceptance target within this budget and had stronger early convergence. At 1,000 additional updates, its mean cosine was 0.98482, compared with 0.96505 for waveform plus mel. The combined objective also improved substantially, so the result does not establish that mel training is fundamentally wrong or that the earlier architecture lacked capacity. Its lower mel error is a reason to retain spectral training later.

The comparison also changes the learning-rate schedule and adds updates relative to the old 500-step run. Therefore, improvement over that old run cannot be attributed exclusively to one of those factors. The active-share renormalization is part of the waveform-only intervention, not a perfectly isolated test of mel-gradient direction.

## Remaining quality gap

The separate unseen sentinel set still fails. For waveform-only, mean nonquiet waveform cosine is **0.30565**, and no nonquiet sentinel crop meets the amplitude requirement. Its quiet sentinel RMS is **0.007715**, versus **0.000121** for the teacher. The waveform-plus-mel arm also has poor sentinel reconstruction.

Those are material failures. They cannot be described as teacher-quality audio, and a fitted diagnostic pass must not be substituted for a held-out quality pass. These examples were deliberately excluded from optimization and came from a different language/source mix. Representative training is the next test of generalization; its success is not guaranteed by memorizing the diagnostic set.

## Chosen next direction

Keep the full decoder architecture and frozen encoder/teacher. Use a **waveform-led training curriculum** as the next representative-training candidate: establish sample reconstruction first, then introduce mel gradually while watching actual gradient contributions, amplitude and noise. Add the implemented adversarial and feature-matching components after stable reconstruction, with an explicit stage transition and checkpoint rollback on regression.

Use a fresh student for the prepared 20.007-hour unused-data pilot. The intentionally fitted diagnostic weights will not seed that run. Main-data exposure must remain nonrepeating and audited. Evaluate held-out multilingual and expressive data; retain the 0.99 reconstruction goal and separate perceptual acceptance requirements.

Before that launch, the representative-data runner still needs the selected curriculum, finite-data cursor and checkpointed stage transitions. A small diagnostic's fixed normalization statistics and moving-average history must not be copied into fresh varied-data training as if they were universally calibrated. No automatic pilot or adversarial run was launched by this comparison.

Both final one-thread CPU checks passed for normalization folding and streaming over eight chunks and 30,720 samples, with maximum streaming differences below 0.000000373. This verifies that limited parity check only. No new RTF, PESQ, STOI, UTMOS, DNSMOS or human-listening benchmark was performed.

The code remains on `Convnext`, uncommitted. See the [protocol](protocol.md), [compact measurements](metrics-summary.json), [full saved results](summary.json), [correctness checks](checks.json) and [TensorBoard curves](https://34d6pb4ub5ldrz-8888.proxy.runpod.net/#scalars&tagFilter=nonquiet_cosine).
