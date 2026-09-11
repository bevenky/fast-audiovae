# Two-native-direction comparison amendment

The original three-direction experiment stopped at its predeclared calibration gate. Three calibration batches ran; none of the four comparison batches ran. The retained checkpoint and optimizer state are unchanged.

The plain-gradient reference failed the quiet-error curvature check throughout the eight-radius grid. This does not establish that plain gradient descent cannot work at any learning rate. It prevents claiming that all three methods have a qualified common displacement in this experiment.

Both native directions, with retained and freshly initialized generator optimizer state, passed every calibration batch at parameter-displacement radius **0.0026359612391104728**, or **1/16 of the initial calibration radius**. This is the largest tested radius that qualifies both methods.

Before evaluating any comparison-batch output, the scope is amended to:

1. Preserve and pin the original report and its deterministic source selection.
2. Compare only those two qualified directions at that fixed radius on the original four comparison batches.
3. Include the original unscaled retained update as a separately labeled reference.
4. Keep the plain-gradient reference marked unqualified. Do not extend its grid or infer superiority from excluding it.
5. Preserve the same checkpoint, teacher targets, discriminator update, raw gradient, loss balancing, source exclusions and ten diagnostic probes. No long continuation, state reset or checkpoint promotion is authorized by the script.

The amendment uses training calibration evidence only. It does not choose a radius or method using the diagnostic panel. It can test sensitivity to generator optimizer history; it cannot isolate AdamW from Muon or establish sustained-training quality from single updates.

[Original experiment report](optimizer-comparison.json) · [Original protocol](protocol.md)
