# Quiet and low-volume audio: findings and corrections

The low quiet-audio score is a real reconstruction failure. Low amplitude by itself does not reduce cosine: scaling both waveforms by the same positive factor leaves their cosine unchanged. Exactly silent signals require another metric, but the recorded quiet example is not a numerical-epsilon case.

| Waveform-only example | Cosine | Student level versus teacher |
| --- | ---: | ---: |
| Fitted quiet expressive clip | 0.9933 | +0.079 dB |
| Fitted low-volume German clip | 0.9975 | +0.020 dB |
| Unseen nearly silent LibriSpeech interval | 0.1560 | **+36.064 dB** |

For the unseen quiet interval, teacher RMS is 0.000121 and student RMS is 0.007715, a factor of 63.6. That is approximately -78.3 versus -42.3 dBFS. Only 0.0004% of student signal power comes from its constant offset. Removing that offset barely changes the error. In 29 of its 32 short windows, teacher RMS is below -80 dBFS while the student continues to produce substantial output. These are measurements of saved waveforms, not new inference or training runs. [Numerical audio analysis](quiet-output-analysis.json).

## Why this remains weak

First, the fitted diagnostic does not cover very quiet conditions adequately. Across its 1,024 twenty-millisecond windows, only five windows total 0.10 seconds below -80 dBFS; none falls below -100 dBFS. The unseen set contains 61 windows, or 1.22 seconds, below -80 dBFS, including 19 below -100 dBFS. That is a coverage difference, not proof of a single architectural cause. [Coverage measurements](quiet-window-coverage.jsonl).

Second, whole-clip metrics and losses can hide errors in pauses. A loud syllable can dominate waveform cosine while low-energy intervals remain inaccurate. The existing normalized waveform loss already increases the weight of low-volume clips, but its normalization uses the complete scored crop. Quiet intervals inside louder clips can still receive comparatively little attention.

Third, the old quiet acceptance rule was incomplete. It allowed student RMS up to the larger of 0.001 and the teacher's RMS plus 1 dB. For this unseen example, that would allow an 8.24-fold increase, or +18.3 dB. An RMS-only rule also cannot reject a muted breath, inverted waveform or unrelated signal with similar energy. The actual current student still failed the old rule, so this did not create a false pass for that unseen example.

The problem is broader than quiet audio: all 31 louder held-out crops also fail the amplitude check. Passing on fitted examples establishes learnability, not generalization to unseen sources and languages.

## Implemented preparation

A new standalone `quiet_audio` module now measures every valid 20 ms window, including partial tails. It uses the detached teacher to identify quiet windows, excludes context and padding, and checks both residual waveform RMS and excess output amplitude. It does not require cosine for silence.

The proposed residual limit is the larger of 0.00001 RMS and 14.14% of teacher RMS; the output-amplitude ceiling is the larger of 0.00001 and teacher RMS plus 1 dB. These are explicit provisional reconstruction tolerances, not calibrated audibility standards. They are intended to expose errors for review rather than redefine a historical result silently.

Applied retrospectively to saved outputs, these stricter limits reveal errors within otherwise high-correlation fitted clips:

| New strict quiet-window check | Waveform plus mel | Waveform only |
| --- | ---: | ---: |
| Fitted quiet windows failing | 92 / 123 | 103 / 123 |
| Unseen quiet windows failing | 123 / 123 | 123 / 123 |

These counts are short windows, not recordings. They do not mean every flagged window is perceptually unacceptable. They do show why the earlier clip-wide 0.99 diagnostic is insufficient for final quiet-audio acceptance. The waveform-plus-mel arm performed better on this finer check, reinforcing that waveform-only is an initial training stage rather than the complete proposed recipe. [Per-window evidence](quiet-region-checks.json).

The module also implements a proposed training term: teacher-relative residual error, averaged over quiet windows within each example and then over examples containing quiet windows. The same 0.001 normalization floor caps near-silence gradients; it was not blindly lowered. The term preserves the teacher's actual room tone and breathing because the target remains the original teacher waveform.

Nineteen CPU tests passed. They cover finite gradients at exact matches, detached teacher targets, injected noise, muting and polarity inversion, masked context and padding, partial tails, and batches with no quiet windows. This loss is **not yet connected to the trainer or used to retrain a checkpoint**. The model's quiet-output failure has not yet been fixed by training.

## Next training changes

1. Add genuine quiet speech, room tone, pauses and speech-to-pause transitions from unused training sources, retaining their original causal context. Stratify coverage by measured level before scheduling the representative pilot. The failing held-out examples remain excluded from training.
2. Include the quiet-window term with a separately logged, bounded contribution. Calibrate it on training-side examples so it improves local residuals without weakening speech reconstruction. Keep ordinary waveform and mel supervision and introduce the planned perceptual losses gradually.
3. Include quiet-window residual and output-energy results in validation alongside whole-clip correlation. Calibrate the absolute tolerance with held-out listening and the declared recording/output conditions; never treat silence or low-level speech as correct solely because output energy is small.

These changes add training and validation work. They add no proposed operations to the CPU inference decoder. The current decoder architecture and latent interface are preserved.
