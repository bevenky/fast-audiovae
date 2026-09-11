# Initial identical-teacher results

The original-weight, unpruned student passes the complete current development evaluation exactly. This is an intermediate review: the 60,000-source cache audit and subsequent bounded optimizer check have not yet completed.

The independent model copy has 191 state tensors identical to the original teacher and no shared tensor storage. Both state hashes are `863109cec1a3cb1f17a90c9d2062dc7f781ee1362069d562567570970ffc07a2`.

All 96 recordings pass native-decoder versus student and native-decoder versus helper comparisons exactly, in 192 train/eval-mode checks. The compared stage boundaries and waveform outputs have zero maximum error. The native teacher and copied student also exactly match the immutable cached targets on scored samples. The current evaluator reports waveform MAE, mel error and full-group MSE all zero, and active cosine exactly one, for both model paths. Their peak is 0.994021, with no overshoots.

| Unchanged quiet cohort | Pruned student at 5,000 | Original-weight copy |
|---|---:|---:|
| All quiet | 1511/2544 | 2544/2544 |
| Near-silence | 171/184 | 184/184 |
| Startup first 20 ms | 0/13 | 13/13 |
| Source-silence teacher transient, 20 to 40 ms | 10/10 | 10/10 |
| Sustained source silence after 40 ms | 164/164 | 164/164 |
| Interior near-silence after 800 ms | 50/50 | 50/50 |
| Quiet with nonzero original source | 1337/2359 | 2359/2359 |

Every copy cohort has zero residual and zero output-limit excess. Its output RMS equals teacher RMS. The tensor-only cached-target self-check also passes on the same GPU. All paths preserve the existing window identity `b74504a0ed422cc106510873b9a270da0d32c4b4b8dd34bd9f31858fb0753660`. Cohorts overlap; the table does not represent disjoint counts.

The no-update training-path probe is complete on twelve fitting sources. With gradients enabled, all three actual reconstruction losses are zero, all 90 parameter gradients are finite zero, and group and waveform outputs exactly match no-grad execution, the native teacher and cached targets. This probe performs no optimizer update.

## First-call qualification

One of the 54 recorded shape warmups is nonexact: shape `[1,64,94]`, source `hi_in:train:17472433744685771543.wav`. Both the first-versus-third no-grad output and first-versus-third grad-enabled output differ slightly. The largest difference, 2.861e-6, is in the hidden group output. For first-versus-third grad-enabled waveform output, maximum difference is 2.682e-7 and RMS difference is 2.666e-8. First grad-enabled output versus the third no-grad output has the same discrepancy.

This is measured startup rounding in this isolated full-width execution. The warmed full-panel checks and twelve-source gradient probe are exact. It does not establish a persistent grad-mode bias, reproduce historical backend state, or demonstrate that the narrower student's remaining errors were caused by startup rounding. Because waveform L1 can produce a sign gradient from tiny nonzero discrepancies, first-call behavior remains relevant to interpreting an optimizer stability test and must not be hidden under the broad compatibility tolerance.

At the inspected saved progress snapshot, 1,500 of 60,000 consumed-source crops had been audited with zero native/copy discrepancies, zero valid-cache discrepancies and zero quiet failures. That snapshot is not a claim that the full scan has finished.

The results establish that the current evaluator recognizes exact original-teacher reconstruction, including startup silence, on the entire development panel. They do not calibrate the quiet thresholds to audibility, prove that the pruned architecture can attain the teacher function, or establish zero-gradient behavior for every training crop before the remaining audit completes. Continue the already approved identity control; no further training or pruning decision is inferred from this intermediate result.

Evidence was summarized by read-only remote calculations from `initial-state.json`, `cached-self-baseline.json`, `development-forward-identity.jsonl.gz`, `native-teacher-before.json`, `control-before.json`, `grad-enabled-pre-update.json`, `warmup-parity.jsonl.gz` and the saved progress snapshot. Raw reports remain on Runpod. This independent review executed no models or optimizer steps.
