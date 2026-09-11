# Training exposure and loss-accounting audit

The completed continuation did train on quiet audio. Near-silence was a much smaller and unevenly distributed subgroup. The loss code includes both; this audit found no quiet-mask exclusion or missing-data bug.

This was a read-only CPU scan of the valid teacher waveform targets from all 40 sealed shards, fresh source indices 12000:24000. All shard hashes matched their receipts. All 12,000 unique source IDs, 1,000 update cursors and 1,409,801,982 scored samples matched the completed training journal. No model was imported or run, and no training state changed. The scan and current corpus inventory took 19.8 seconds. Full statistics are in [training-exposure-audit.json](training-exposure-audit.json); the standalone reader is [audit_training_exposure.py](audit_training_exposure.py).

| Teacher-defined region | Latest training exposure | Training sample share | Fixed development sample share |
| --- | ---: | ---: | ---: |
| All scored audio | 8.159 hours | 100% | 100% |
| Quiet, 20 ms RMS ≤ 0.001 | 73.45 minutes | 15.004% | 21.668% |
| Near-silence, RMS ≤ 0.00001 | 3.96 minutes | 0.809% | 1.564% |

Near-silence is a subset of quiet. Counts use contiguous 960-sample windows within each scored crop and include its final partial window with the actual valid length. Context is available to the causal decoder but excluded from every scored denominator, as intended.

Every one of the 1,000 updates contained some quiet samples. Near-silence appeared in 1,087 of the 12,000 source crops and 670 of the 1,000 updates; **330 updates contained none**. Across updates, its median sample share was only 0.204%, versus a mean of 0.811%. A source with any near-silence typically contributed 0.10 seconds of it. There were 43 entirely quiet crops, but no entirely near-silent crops. No cached teacher window was exactly zero; this does not establish that input recordings lacked silence, because the target is the teacher's reconstruction.

Near-silence exposure was 0.797%, 0.787%, 0.950% and 0.703% in successive 250-update blocks. The final block had the least near-silence exposure while the development near-silence residual improved, so these counts alone do not explain the trajectory causally. Startup crops contained 1.367% near-silence; crops with preceding context contained 0.313%. Thus coverage of near-silence inside established causal history is especially limited.

## What the objective actually weights

The unchanged update uses raw sample-pooled waveform L1, five-scale mel loss, and complete stage-4 boundary MSE. Coefficients remain 1, 0.0006674012905982311 and 0.009304078923434964. For all 1,000 saved updates, the weighted branches sum exactly to the recorded total.

- **Waveform:** `sum(abs(student − teacher)) / valid_samples`. Quiet and active samples have the same derivative magnitude `1 / valid_samples` when their error is nonzero. There is no inverse-RMS normalization, silence mask, or separate per-clip weighting. A small absolute residual contributes a small loss value, but it does not receive a smaller L1 derivative merely because its amplitude is low. Near-silence nevertheless supplies fewer sample terms because it is only 0.809% of exposure.
- **Mel:** all complete, valid STFT windows enter the loss, including quiet windows. The natural-log branch clamps mel magnitudes to 1e-5. A student bin below that floor has zero derivative through that log branch; waveform and linear-mel supervision still remain. This mel threshold is not a waveform RMS threshold. The current reports do not measure how many failing near-silence bins are below the spectral floor, so this is a possible sensitivity limit, not a demonstrated cause.
- **Stage-4 features:** every valid sample weights its corresponding 12 kHz feature cell, including partial cells. All 128 channels are compared. Quiet waveform amplitude does not imply a zero teacher feature vector. No activity-dependent feature normalization or special near-silence weighting exists.

The weighted scalar objective was about 94% waveform by value. That percentage is **not** a gradient-share measurement: the mel and feature coefficients were calibrated by parameter gradients, and their current regional gradient contributions are not logged. These saved numbers cannot establish that one branch dominates near-silence updates or causes its residual.

The evidence supports **limited near-silence exposure and a global objective without a special relative-noise requirement**, rather than missing quiet training data. It does not prove that more ordinary speech alone will repair the noise floor, nor that the architecture has reached a capacity floor. A new download is unnecessary to address the coverage question; existing recordings can supply a deliberately measured future source mix, subject to preserving the no-repeat ledger.

## Data needed to reach optimizer step 10,000

The current checkpoint is optimizer step 5,625. At accumulation 12, reaching 10,000 requires **4,375 more updates and 52,500 new distinct source crops**. Only 3,000 sources remain in the existing 27,000-source plan, so an additional 49,500 sources would need a new sealed selection and teacher cache. Counters cannot be inferred as `step × 12`, because the earlier run used accumulation 3.

A current read-only inventory of the existing master corpus found **151,147 unselected eligible training records, representing 421.709 manifest hours**. All 151,147 audio paths still exist and are nonempty. This excludes the plan's reserved identities plus all 27,000 already planned sources, deduplicates source ID, declared audio hash and parent recording, and requires the current minimum scored length. The master SHA remains `198ed7aa345db0675c8db3382b9856a31857a66b6a17c62ed9d581d781f5c8fd`.

There is enough existing audio to select the additional 49,500 unique sources without another download. This is a metadata/existence inventory, not a new audio-byte validation or a completed teacher cache. Missing speaker/session identities cannot establish speaker-disjointness. Available disk remains a practical preparation constraint: about 544 MiB on `/workspace`, 3.8 GiB on `/tmp`, and 25 GiB of volatile `/dev/shm` at audit time. No new selection, deletion, cache production or continuation was started by this audit.
