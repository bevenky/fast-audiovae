# Independent assessment of targeted data and discriminator views

**Recommendation: keep this training-data/view package provisionally.** It improves the intended expressive and quiet outcomes without an aggregate ordinary-speech reconstruction penalty. It does not establish uniform perceptual quality, solve silence, or control every overshoot.

The following comparisons use all four completed, verified 400-update trials in `outputs/convnext-corrected-experiments/results.json`. The targeted result is compared with the matched regular-speech control; both start at step 8,090 and receive exactly matched scored audio duration per batch position.

| Metric | Targeted versus regular | Targeted versus step 8,090 |
| --- | ---: | ---: |
| Natural quiet pooled residual RMS | 29.03% lower | 11.38% lower |
| Natural 480-sample residual-template median | 49.93% lower | 28.25% lower |
| Expressive waveform MAE | 2.96% lower | 3.83% lower |
| Expressive fixed diagnostic mel | 3.64% lower | 2.92% lower |
| Speech waveform MAE | 0.04% lower | 0.74% lower |
| Speech fixed diagnostic mel | 1.78% lower | 1.46% lower |

Expressive nonquiet correlation rises from 0.7511 in the regular control to 0.7734 with targeted training, compared with 0.7447 in the parent. Speech nonquiet correlation is effectively unchanged versus regular: 0.95614 to 0.95621. These correlations must not be compared with quiet waveforms whose cosine is undefined or unstable.

The quiet gain is more than mitigation of the regular continuation's drift: targeted improves against the parent too. Nevertheless, all **3,434 natural teacher-quiet windows still fail** the current engineering checks. The residual-template statistic includes DC and is not proof that a particular layer causes the artifact.

The principal adverse result is laughter overshoot. The expressive maximum rises from 1.1335 to 1.1675 versus regular, with 95 to 119 samples above full scale in the same one source and two overlapping crops. It is also worse than the parent peak, 1.1390. Across all natural clips, peak severity is lower than regular but slightly higher than the parent. Do not describe this package as a peak-control fix.

Natural waveform MAE improves in 191 crops and worsens in 91; fixed mel improves in 219 and worsens in 63. The source-bootstrap interval for the natural waveform-MAE change versus regular is approximately −1.93% to −0.45%. It describes variation across these held-out sources, not training-seed uncertainty, speaker clustering, or perceptual equivalence.

Some small speech-language cohorts regress in waveform MAE versus regular: Tamil 1.75%, Telugu 1.67%, and Sindhi 1.24%, each represented by three sources. Their absolute changes are small and quiet generally improves, but aggregate improvements do not justify claiming every language improved.

This intervention changes both source selection and the discriminator's 190 ms view. It cannot isolate their individual benefit. The control contains 99.630% speech rather than the original broad-emotion mixture. Explicit labels are source-level and activity-qualified, not semantic event timestamps. Training contains all 22 Indic languages, but separate held-out crying, giggling and reviewed shouting cohorts remain missing; the additional Yell recording is one diagnostic only. The final all-event summary should be consulted before making claims about a particular expressive category.
