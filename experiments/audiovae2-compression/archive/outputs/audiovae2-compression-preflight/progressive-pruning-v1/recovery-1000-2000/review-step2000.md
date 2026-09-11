# Same-width recovery completed at update 2,000

Retain the checkpoint and stop awaiting review. This is a successful continuation at widths 384/256, not a quality-qualified model release. No next cut has started.

Checkpoint SHA256: `a7b6c5ee850c80061c073dd42ab7a019ce96d55b0b9691b77c9721ea0b14602f`.

| Held-out metric | Update 1,000 | Update 1,500 | Update 2,000 |
| --- | ---: | ---: | ---: |
| Active waveform cosine | 0.988697 | 0.991630 | 0.991865 |
| Waveform MAE | 0.003286 | 0.002671 | 0.002652 |
| Mel error | 0.201765 | 0.173676 | 0.160883 |
| Complete-group MSE | 0.003539 | 0.002741 | 0.002404 |
| Active pooled RMS / teacher | 93.65% | 97.90% | 99.95% |
| Quiet windows passing | 358/2544 | 1000/2544 | 1066/2544 |
| Near-silence passing | 167/184 | 167/184 | 171/184 |
| Sustained source silence passing | 164/164 | 161/164 | 164/164 |
| Interior near-silence passing | 49/50 | 46/50 | 50/50 |
| Startup near-silence passing | 0/13 | 0/13 | 0/13 |

The same 96 held-out recordings were evaluated throughout. Relative to update 1,000, waveform MAE improved on 95/96 recordings, and all 96 improved on mel and complete-group MSE. Aggregate MAE improved 19.29%, mel error 20.26%, group MSE 32.07%, and quiet residual RMS 21.46%. There were no full-scale overshoots; the maximum absolute output was 0.989645.

All remaining 13 near-silence failures are in the first 20 ms. Quiet audio with a nonzero teacher signal also remains imperfect. The quiet cohorts below overlap and must not be summed.

| Quiet cohort | Residual RMS at 1,000 | At 1,500 | At 2,000 |
| --- | ---: | ---: | ---: |
| All quiet | 111.87 | 91.73 | 87.87 |
| Near silence | 6.47 | 5.64 | 5.50 |
| Startup, first 20 ms | 19.85 | 20.37 | 19.76 |
| Teacher transient, 20 to 40 ms | 289.57 | 233.93 | 158.80 |
| Source silence after 40 ms | 3.54 | 1.45 | 1.41 |
| Interior near silence | 3.52 | 1.83 | 1.52 |
| Quiet nonzero reference | 114.61 | 94.01 | 90.64 |

Residual RMS is expressed in millionths of full scale. Startup error barely changed. Sustained and interior silence recovered their lost amplitude-threshold passes without special silence processing.

The last 500 updates were less uniform: 54 recordings improved MAE and 42 worsened. Kannada, Nepali, Sanskrit, a whisper and a yell are among the regressions to retain on the comparison panel. Pooled RMS matching does not establish per-recording amplitude parity. Whistle 428921 recovered its shape regression, finishing with cosine 0.994777 and MAE 0.000488416, but its RMS is still 92.25% of the teacher. Do not infer a plateau from the last interval's 0.69% aggregate MAE gain: mel still improved 7.37% and group MSE 12.30%.

The boundary review shows complete stage-4 NRMSE improving from 0.17553 to 0.15533 to 0.14361; final waveform NRMSE on that panel improved from 0.18079 to 0.13866 to 0.13831. Quiet intermediate representations remain imperfect despite better final quiet output. Internal distance by itself is not a recovery gate.

Integrity checks passed: all 90 AdamW states reached update 2,000 without a reset; coefficients, widths and optimizer settings were unchanged; weights, moments and losses were finite; 12,000 new distinct sources followed the approved order; all 12,000 teacher/cache comparisons passed; and frozen state and protected files were preserved. The final checkpoint hash was independently verified. TensorBoard exposes all 15 overview curves through update 2,000. Raw reports and integrity evidence remain on Runpod.

Recommendation: a separate controlled trial of the planned next width cut, 384/192, is defensible, retaining this checkpoint as the control and the original AudioVAE2 as teacher. This requires its own go-ahead and source preparation. The original 30,000-source stream has 6,000 unconsumed sources left, short of the 12,000 needed for another 1,000 updates. Extend disjoint preparation before a full next cut. Continue separate silence, amplitude, per-recording and boundary measurements from the first update. Correlation is not perceptual equivalence, and this review measured no CPU RTF.
