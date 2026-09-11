# Final checkpoint: bounded waveform diagnostic

We analyzed 12 development crops from the completed 1,009-update checkpoint, covering English, Hindi, Latin American Spanish, French, two quiet expressive clips, laughter, screaming, whistling and another expressive clip. This used frozen teacher/student inference only. No training, waveform correction or performance benchmark was run. All audio remained on Runpod.

## A shared repeating component appears in quiet output

The decoder emits 480 samples for each internal frame. Folding its quiet output into 480-sample blocks exposes a similar repeated waveform across unrelated recordings. The strongest evidence comes from crops containing many quiet windows, avoiding the misleadingly high fit that two blocks alone can produce.

| Quiet development crop | Quiet 20 ms windows | Student AC power explained by repeating template | Teacher equivalent | Student template RMS |
|---|---:|---:|---:|---:|
| Expressive 000216 | 26 | 51.9% | 4.34% | 0.000753 |
| Expressive 000298 | 24 | 29.7% | 0.64% | 0.000755 |
| Whistling 457967 | 36 | 29.0% | 1.17% | 0.000740 |

The three student templates have cross-recording cosine similarity of 0.891–0.918. Their RMS is about -62.5 dBFS. DC explains only 0.018–0.409% of the student quiet power in these examples.

This is evidence of a shared component repeating every 10 ms, consistent with the 480-sample waveform head. It is not proof that one particular bias or layer causes it. A direct waveform head can turn constant intermediate features into a repeated output vector even when its final convolution has no bias. Removing DC alone would not address the problem, and substantial residual error remains after diagnostically subtracting the repeated template.

No output was actually modified. A useful additional monitoring statistic is the teacher-relative 480-phase residual template on sufficiently long quiet intervals. Any targeted training penalty would need a separate controlled check so it does not suppress real periodic vocal sounds. Hard muting, post-hoc template subtraction and removal of learned biases are not validated fixes.

## The decoder also under-reconstructs active sounds

| Selected crop | Unshifted cosine | Student level relative to teacher | Best diagnostic lag | Cosine after that diagnostic shift |
|---|---:|---:|---:|---:|
| English speech | 0.513 | -8.72 dB | 0 samples | 0.513 |
| Hindi speech | 0.740 | -9.99 dB | 0 samples | 0.740 |
| Spanish speech | 0.667 | -11.43 dB | 2 samples | 0.669 |
| French speech | 0.641 | -4.47 dB | 2 samples | 0.644 |
| Median selected laughter | 0.407 | -13.14 dB | 11 samples | 0.440 |
| Screaming | 0.0028 | -20.74 dB | -810 samples | 0.018 |
| Whistling 457967 | 0.0358 | -22.29 dB | 485 samples | 0.073 |
| Whistling 539498 | 0.0986 | -19.73 dB | 157 samples | 0.140 |

The bounded lag search covered +/-20 ms. The results do not support a single global sample-offset error as the cause. Gain correction alone also cannot improve cosine, and the worst nonverbals have very little aligned teacher signal in the output.

For the four speech examples, energy from 300–3,000 Hz is 9.7–14.6 dB below the teacher. Low-frequency waveform correlation below 300 Hz is 0.886–0.920, while 3–8 kHz correlation is only 0.019–0.059. The student has learned some coarse voiced structure but has not learned much of the detailed waveform. Relative excess in upper bands must be read alongside absolute RMS, because the teacher has very little energy there on several clips.

## Provenance and limits

The original bounded training cache had evicted all development targets by the end of the run. We regenerated only the selected complete utterances using the pinned, frozen original teacher in deterministic FP32 with the recorded warmup policy, then selected the exact original crops and causal context. No training asset or cache index was changed.

The source identities and crop positions match the checkpoint. Two of 12 latent tensors and one of 12 teacher tensors match the original saved bytes. Other byte differences are explicitly recorded in the JSON, as serial target regeneration can differ from the earlier qualified mixed-length batching. For 11 crops, cosine differs from the original saved scalar by less than 0.0000002; one laughter crop differs by 0.000455. The maximum relative student-RMS difference is 0.509%, and the maximum teacher-RMS difference is 0.719%. These diagnostics do not replace the original acceptance results.

This is a deliberately small, selected panel totaling 21.12 scored seconds. It identifies mechanisms to investigate, not population-level quality or a definitive attribution to the output head.

Checkpoint bytes, training source archive, training metrics, cache index, student state and teacher state were unchanged after the diagnostic. See [full per-crop measurements](waveform-audit.json) and the [recipe audit](recipe-audit.md).
