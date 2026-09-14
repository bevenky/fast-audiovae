# Accepted AMD INT8 baseline

The accepted experimental reference is **0.137536 RTF**: the median of three paired rounds on AMD EPYC 9654, one CPU thread, causal 80 ms decoder packets and ONNX Runtime 1.30.0. The pooled value is 0.137168. These statistics describe the same run and should not be mixed when comparing later results.

[80ms.json](80ms.json) is an exact copy of the measured combined graph specification. [manifest.json](manifest.json) pins its graph, configuration and evidence hashes, along with the quality scope. It does not promote the candidate into the package's default recipe.

Only nominal 80 ms streaming has an accepted performance reference here. Keep this graph for the entire stream, including a final 40 ms packet; do not exchange its state with another graph. The copied specification also lists 40 ms correctness cases, which do not establish a 40 ms timing baseline.

The six full-utterance quality pairs preserved existing INT8 waveform samples exactly. The existing quantization still has a measured quality cost versus FP32, including a mean PESQ decrease of 0.04651 on this small panel. See the manifest and [quality audit](../../quality-audit.md) before making a broader claim.

From the experiment directory, `python resolve_baseline.py --packet-ms 80` verifies the local manifest/specification pins and prints the original configuration without inference. On the AMD host, `--verify-artifacts` additionally verifies the recorded runtime files. Restoring those files is separate from resolving the configuration.
