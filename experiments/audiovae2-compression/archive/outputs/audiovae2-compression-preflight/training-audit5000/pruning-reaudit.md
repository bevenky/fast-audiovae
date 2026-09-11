# What was pruned from the original AudioVAE2 decoder

The student retains the original decoder's module families and temporal structure. It narrows channels; it does not currently remove residual units. This preserves the type of architecture while changing its capacity and learned function.

| Stage | Teacher input/output channels | Student input/output channels | Residual units |
|---|---|---|---|
| 2 | 1,024 / 512 | 1,024 / 256 | All three retained |
| 3 | 512 / 256 | 256 / 128 | All three retained |
| 4 | 256 / 128 | 128 / 128 | All three retained |

The original encoder, stem, stage 1, stages 5–6 and output head are frozen. Snake, depthwise temporal convolutions, pointwise channel mixing, residual additions, conditioning, weight normalization and causal upsampling remain present. The final neighboring-sample convolution and tanh are retained too. The student is constructed using the original stage class, with different widths and consistently selected parameters.

## Correct connections do not imply an unchanged function

Consider the first residual unit of stage 2 at the original pruned initialization. The stage's input is unchanged. Its upsampler's retained output channels therefore initially reproduce the corresponding teacher channels. The following Snake and depthwise convolution operate separately per channel and preserve that selected-coordinate correspondence, subject to floating-point evaluation.

The first pointwise convolution mixes all teacher channels. For retained output coordinates K and removed coordinates D, its teacher output is:

```text
teacher_K = W_KK * features_K + W_KD * features_D + bias_K
student_K = W_KK * features_K                         + bias_K
```

The initialization discards the second contribution. This follows directly from `group_model.py` selecting both input and output axes of the pointwise weight. It is intentional channel pruning, not a missing residual skip or an incorrect convolution axis. Later pointwise and upsampling convolutions also lose incoming contributions from removed channels.

Those contributions can reinforce or cancel others. A channel can be redundant enough to approximate from retained channels while still contributing nonzero values through its outgoing weights. Selecting channels alone does not automatically fold those contributions into the retained weights. The current initializer slices the effective weights and reconstructs their weight-normalization parameters, then relies on joint distillation to recover the complete stage-2-to-4 function. It does not perform a separate least-squares weight reconstruction before training.

This distinction is established in structured-pruning literature. [He et al., Channel Pruning, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html) combines channel selection with least-squares reconstruction. [ThiNet, ICCV 2017](https://www.lamda.nju.edu.cn/luojh/project/ThiNet_ICCV17/ThiNet_ICCV17.html) uses the effect on the next layer when selecting channels. These are methodological precedents, not audio-quality results or evidence that a particular reconstruction method will fix this student.

## What our checks rule out

The source and numerical audits found no mismatched selected-channel axes, missing selected bias/Snake parameters, stale normalization scale, duplicated or missing sample-rate conditioning, changed stride/dilation/padding, or detached gradient through the frozen suffix. The full-width copied control passed before compression. On the newly inspected sources, teacher group features still yield bitwise teacher audio through the student's unchanged suffix.

The loss matches the complete 128-channel output of stages 2–4. It does not force the narrower internal stages to reproduce the full wider teacher representation. Partial internal coordinate diagnostics are explicitly labelled and are not extra training losses.

## What remains unproven

- Halving both internal widths in one operation has not been shown to retain enough capacity for the final reconstruction target on every speech, silence and expressive case.
- Activation-based channel selection is not proof that the removed outgoing contributions were unimportant to final audio, particularly when hidden features cancel to produce a very small waveform.
- The missing reconstruction initializer may make initial recovery harder, but it does not by itself explain why the model regressed between steps 4,500 and 5,000, long after pruning occurred. That interval needs the separate exact replay.
- Matching average group MSE more closely does not guarantee lower waveform error. Different feature-error directions have different effects through the fixed nonlinear output stages.

Do not remove additional residual units or add a new output architecture on the strength of these observations. First identify the late update behavior. If subsequent evidence identifies a pruning-capacity or initialization limitation, compare a reconstruction-aware initializer or a staged width reduction under matched data exposure. Those are conditional experiments, not changes made in this audit. Neither requires changing the frozen original teacher.
