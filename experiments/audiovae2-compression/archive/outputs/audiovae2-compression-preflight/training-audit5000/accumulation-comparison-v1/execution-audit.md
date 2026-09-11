# Accumulation comparison execution audit

The approved comparison completed successfully. Main training remains paused at its original step 5,000. Neither diagnostic checkpoint was promoted.

## What was held constant

Both arms restored the original step-4,500 group parameters, all AdamW moments and counters, and all saved random states exactly. They used the same ordered 1,500 distinct sources at fresh-plan positions 10,500 through 11,999, totaling 176,445,657 scored samples, or 1.02110 hours. Context and padding were excluded from this exposure.

Both used the original singleton forward path, FP32, disabled TF32, learning rate 0.00003, and unchanged waveform, mel and group-boundary losses and coefficients. Loss contributions were pooled over the actual valid sample and spectral-element counts in each effective batch. There was no additional division by accumulation size. Independent CPU tests checked that arithmetic on unequal-length, quiet and active examples against a full-batch oracle.

| Check | Result |
|---|---|
| Independent focused tests | 15 passed locally and on Runpod |
| Common initial weights, optimizer and RNG | Exact for both arms |
| Initial full 96-source reports | Passed existing saved-metric tolerance; paired aggregates exactly equal |
| Sources per arm | Same ordered 1,500, each used once within that arm |
| Teacher versus cached target | Bitwise equal on all scored samples for all 1,500 sources in each arm |
| Trajectory observations | All 26 points at 0, 60, …, 1,500 sources |
| Full validation | Same 96 sources before and after each arm |
| Frozen teacher, prefix and suffix | Preserved |
| Original files and checkpoints | Preserved |
| GPU process after completion | None |

Accumulation of 3 made 500 optimizer updates, ending at AdamW counter 5,000. Accumulation of 12 made 125 updates, ending at counter 4,625. The distinct diagnostic checkpoint format records optimizer count and source exposure separately. It cannot be mistaken for the main trainer's source ledger.

## Endpoint measurements

| Metric | Accumulation 3 | Accumulation 12 |
|---|---:|---:|
| Waveform MAE | 0.005525857 | 0.004234362 |
| Waveform MSE | 0.000302783 | 0.000226950 |
| Mel error | 0.342078894 | 0.335224718 |
| Group-boundary MSE | 0.011688219 | 0.011624735 |
| Mean active waveform cosine | 0.972512973 | 0.976007129 |
| Quiet residual RMS | 0.000176862 | 0.000176647 |
| Failed quiet windows | 2,425 / 2,544 | 2,406 / 2,544 |
| Failed near-silence windows | 180 / 184 | 166 / 184 |
| Full-scale overshoot samples | 0 | 0 |

The larger accumulation has 23.37% lower waveform MAE, 25.05% lower waveform MSE and 2.00% lower mel error at matched final exposure. Quiet residual RMS improves only 0.12%; most quiet and near-silence windows still fail the unchanged thresholds. These results support evaluating the batching policy further, but do not establish that silence has been fixed or the model has reached its quality target.

Actual minibatch gradient dotted with AdamW displacement was positive on 33 of 500 control updates and 2 of 125 larger-batch updates. This is descriptive: momentum can produce such movement, and the gradient is for the current training batch, not the held-out panel.

## Interpretation limits

- This is one fixed source interval and starting checkpoint, not multiple seeds or training periods.
- At equal source exposure, accumulation changes the number of optimizer updates and AdamW's moment history measured in examples. It does not isolate gradient variance alone.
- Both arms use the same new instrumented path. Historical bitwise reproduction of the main run is not claimed.
- Four diagnostic sources are measured throughout; the full 96-source panel is measured at the endpoints. No best intermediate checkpoint was selected.
- No architecture, inference operation, learning rate, loss coefficient or quiet threshold changed. This experiment does not measure CPU RTF.

Runner SHA256: `bb4d5e69da28a4072ab8b3758c5c6149a2a4489f23429a470e2c29cd8d5871f1`.

Test SHA256: `efacffea2391d50bf6dcdb8eccb13b1e69ed113cc610fc055448f4adf01df8a9`.

Full reports and journals are in `results/`. Model checkpoints remain on Runpod at `/tmp/fast-audiovae-accumulation-comparison-v1/accumulation3/final.pt` and `/tmp/fast-audiovae-accumulation-comparison-v1/accumulation12/final.pt`. The run finished in 248.50 seconds, including both arms' diagnostics; this duration is not a training-throughput benchmark.
