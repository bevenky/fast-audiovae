# Startup and quiet reconstruction diagnosis

The user authorized finishing the exact-teacher control and then investigating the current pruned student's startup and quiet failures. This is a read-only model diagnosis. No new training, pruning, loss changes or inference modifications are part of this experiment.

The current model narrows only stage 2 from 512 to 384 channels. Stage 3 remains 256 channels. All nine residual units in stages 2 through 4, their dilations, causal padding, Snake activations and the original output head remain. Earlier experiments with widths 256/128 do not establish the cause in this current model.

Use the original teacher, the saved current-cut step-zero model and the step-5,000 checkpoint. Preserve identical latent values, sample-rate conditioning, causal context, valid samples and fixed evaluation-window identities. Reproduce saved baseline measurements before interpreting interventions. The complete unpruned control must finish before this diagnostic uses the GPU.

Measure all 13 first-20-ms near-silence failures. Include persistent nonzero-quiet examples selected in advance from Punjabi, Bodo, Somali, whistling and whispering, along with passing sustained/interior silence and active portions of the same recordings. Keep waveform offset, centered time-varying error, gain error, transient location and feature-boundary errors distinct. Summaries must use sample-weighted energy and avoid treating overlapping cohorts as disjoint.

At the exact sliced initialization only, decompose missing input-channel contributions and propagated retained-channel changes at four sites: the three stage-2 residual pointwise convolutions and the stage-3 upsampler input. One-site restoration measures a conditional diagnostic effect; restoration of all four is an accounting control that should reproduce the teacher within the established numerical tolerance. These teacher-driven restorations are not deployable repairs or additional trainable layers. Never inject teacher-coordinate values into the adapted step-5,000 model. Compare the adapted model at complete shared boundaries and waveform output instead.

Read the original causal padding, transpose convolution, sample-rate conditioning and initialization code to interpret measured time patterns. Also summarize existing training crop geometry and reconstruction reduction rules: a correct metric pipeline can coexist with insufficient recovery of a rare or weakly weighted behavior. Do not assume this is the cause without evidence.

Require focused CPU tests, real-model full-width and restoration controls, checkpoint/source hashes, native forward parity, finite values, hook cleanup and unchanged model state. Raw reports and tensors remain on Runpod. Local files contain code and derived assessments only. Preserve the original training run and TensorBoard. Finish with an evidence-backed cause assessment and a proposed next step; do not automatically train a fix.
