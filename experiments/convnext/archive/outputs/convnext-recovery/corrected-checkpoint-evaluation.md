**Corrected-input checkpoint evaluation**

The existing targeted step-8,490 checkpoint was evaluated without an optimizer update on the same 285 held-out crops from 147 sources. All source bytes, crop geometry, scoring masks and checkpoint hashes were verified. Encoder outputs were regenerated independently with the verified cuDNN bypass; teacher decoding used cuDNN 9.25.1. Historical targets were preserved.

| Metric | Historical inputs and targets | Corrected inputs and targets |
|---|---:|---:|
| Nonquiet correlation | 0.9199126 | 0.92011139 |
| Waveform MAE | 0.012551717 | 0.012544517 |
| Mel error | 1.1353604 | 1.1354562 |
| Pooled quiet residual RMS | 0.00030667711 | 0.00030671538 |
| Maximum absolute output | 1.2655061 | 1.2655067 |

Eight sources had material latent corrections. The other 139 differed only at numerical scale. The same 4,073 teacher-defined quiet windows remain, including 3,434 natural windows; the student still fails the existing quiet criteria on all 3,434 natural windows.

The checkpoint retains its reconstruction quality on the corrected encoder distribution in this development panel. This supports preserving it and does not support attributing the remaining silence residual or overshoot to the encoder bug alone. It does not establish the corruption fraction in historical training, perceptual equivalence, or a release-quality pass.

The earlier step-8,090 parent was also evaluated on corrected inputs: correlation 0.91304, MAE 0.0128069, mel error 1.15627 and pooled quiet residual RMS 0.000346118. The targeted checkpoint still improves reconstruction and quiet residual relative to that parent, while its maximum peak remains worse.
