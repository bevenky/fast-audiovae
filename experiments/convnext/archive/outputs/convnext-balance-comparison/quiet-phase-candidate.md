# Conditional quiet phase candidate

Prepared locally as an isolated training component. It is not imported by the active comparison recipe, uploaded to Runpod, or enabled in a trainer.

API:

```python
from audiovae_student.quiet_phase import (
    QuietPhaseConfig, quiet_phase_loss, quiet_phase_metrics,
)

loss = quiet_phase_loss(student, teacher, valid_mask)
metrics = quiet_phase_metrics(student, teacher, valid_mask)
```

Inputs are matching floating `[batch, 1, samples]` waveforms and an optional Boolean mask of the exact same shape. The default settings use 480-sample periods, complete 960-sample windows, at least eight qualifying windows per example, teacher RMS at most 0.001, and a normalization floor of 0.001.

The loss averages the student-minus-teacher residual at each of the 480 phase positions across qualifying windows. It measures that mean template's RMS, divides by the greater of the floor and selected teacher RMS, then averages active examples equally. At least 16 aligned head blocks contribute to each active example. The teacher is detached. Only complete valid windows on the original tensor grid contribute; causal context, partial tails, mask holes and nonquiet windows receive no gradient. An inactive batch returns a connected zero with finite zero student gradients.

The metrics report complete and teacher-quiet window counts, actually eligible windows, active examples, per-example teacher RMS, residual-template RMS and normalized-template RMS. Inactive examples report `None` for the template estimate rather than a quality pass.

Matching periodic teacher audio, low-level tones and room tone cost zero. Muting that teacher signal remains an error. Nonrepeating noise may cancel in the mean template, intentionally: this is a targeted phase penalty rather than a general silence or noise loss. There is no output filtering, template subtraction, muting or new decoder inference operation.

Twenty focused CPU tests pass: exact silence and zero gradients; detached teacher; coherent versus equal-energy canceling residual; matching quiet periodic tones and room tone; incomplete/short windows; masked NaN/Inf context, holes and padding; per-example reduction; nonquiet exclusion; safe half/bfloat accumulation; and malformed input rejection.

Before any conditional trial, first verify that the 480-sample pattern persists in the selected comparison checkpoint, then measure this loss's current gradient norm against waveform and mel contributions. No coefficient has been selected and the rejected earlier 10% general quiet penalty is not reused. The active comparison's frozen source and recipe remain unchanged.
