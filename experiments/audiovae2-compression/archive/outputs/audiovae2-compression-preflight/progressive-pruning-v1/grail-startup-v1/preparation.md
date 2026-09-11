# Fresh G plus startup preparation

11 September 2026. The corrected retention pilot now qualifies the approved fresh combination: all 64 updates moved, all six training startup anchors stayed feasible, development retained 12/13 starts, and waveform MAE was effectively equal to ordinary recovery. Remaining startup residual and 20–40 ms transient errors are still separate review items.

The new initializer starts from the original AudioVAE2 teacher, installs the original untrained G native fits and recalibrates only the existing stage-3 upsampler on actual G input features. Its three G residual mixers remain unchanged. No trained G or retention checkpoint, saved pruned group, extra decoder layer or inference branch is used.

The six G artifact hashes in [config.json](config.json) were read and verified directly on Runpod. G's receipt confirms zero neural updates, no historical checkpoint weight installation, a pristine teacher factory and all nine residual units retained. Its complete untrained decoder state hash is `c2499384e2981fa5ffbaddc945209b9cf1a5c02552f3d58d709c47cf07ee3b2c`.

Original selected decoder state: `01fdc44c20b0dad261fbdf7d8138f1cd4917d3df351943a0d93fba1885928a32`. Original teacher decoder state: `863109cec1a3cb1f17a90c9d2062dc7f781ee1362069d562567570970ffc07a2`.

Only aggregate receipt structure, hashes and provenance were exported. Original G weights, hidden maps and source membership remain on Runpod. Source and tests are being prepared; calibration and the paired fresh recovery pilots have not started. The [fixed combination plan](../startup-retention-audit-v1/fresh-grail-combination-plan.md) governs this experiment.
