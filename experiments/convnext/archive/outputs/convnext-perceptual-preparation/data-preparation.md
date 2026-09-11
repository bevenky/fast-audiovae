# Future perceptual-stage data preparation

Prepared locally only. The r7 run and its frozen remote source remain unchanged. No r8 metadata selection, audio acquisition, upload or training has been started.

The planner now accepts `minimum_input_samples`, defaulting to the historical 1,366 samples. Setting it to **3,040 input samples** guarantees at least **9,120 valid output samples** for every crop, including eligible partial tails. This prevents an adversarial-training batch from having no qualifying 0.19-second crop. Below-threshold tails are disclosed in discarded capacity; qualifying tails are retained exactly, with their real sample count and padding mask.

The default synthetic fixture reproduces the exact window and metadata hashes from the frozen r6 source. The loader additionally rejects windows below their declared minimum even when every metadata file has a valid checksum, and rejects conflicting input/output minimum declarations. Thirteen planner tests and two lineage tests pass.

`work/convnext-preflight/prepare_perceptual_data.py` is a standalone future driver. It requires an independently verified SHA256 for `selected-parent.json`; it does not assume which r7 arm wins. It verifies the selected checkpoint, completed r7 source checkpoint, selected arm, full 250-update/8,000-window journal and pinned r7 data plan before adding the r7 source files to the inherited lineage ledger.

Its future output is `/workspace/fast-audiovae-convnext-20260909-r8/data/perceptual-v1`: 32,000 windows for 1,000 updates at batch size 32, with the new 3,040-input-sample minimum. It preserves source/hash/parent and known speaker/session reservations, all 22 required Indic languages, and the current mixture targets subject to actual capacity. Missing rare events are reported rather than replayed. Expected expressive scarcity means the eventual achieved mixture must be read from the generated report; no current r8 hours or event coverage is claimed.

The driver derives parent hashes and the arm from the verified selection file. If the quiet-phase arm is selected, it explicitly records that a phase-to-perceptual engine transition is still required. Data preparation does not remove a phase extension, change model weights or enable GAN training.

Example command after selection and source publication:

```sh
python -B prepare_perceptual_data.py \
  --parent-selection /workspace/fast-audiovae-convnext-20260909-r8/selected-parent.json \
  --selection-sha256 VERIFIED_SELECTION_FILE_SHA256
```

The default source root and output point to r8. `--dry-run` provides a read-only preview after the selected-parent checks succeed. The published report records actual capacity, absent event classes, achieved fractions and exact lineage provenance, while console output stays compact.
