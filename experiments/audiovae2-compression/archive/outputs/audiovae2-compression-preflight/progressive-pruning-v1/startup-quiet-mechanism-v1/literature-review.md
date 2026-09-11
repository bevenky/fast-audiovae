# Distillation and gradual channel reduction

Gradual reduction with recovery is a reasonable strategy. The important addition is to reconstruct what the retained channels must produce, rather than relying on more training after plain slicing. Published evidence supports the general method; it does not guarantee that this width can reproduce every AudioVAE2 detail.

## What StreamCodec2 actually establishes

**StreamCodec2** uses projected intermediate guidance, including upsampling, alongside reconstruction objectives. Its direct route beats its intermediate-teacher routes but still trails its teacher. Training used900,000 updates. These results do not test our fixed-teacher progressive cuts. [Paper](https://arxiv.org/html/2509.13670v1)

## The closer methodological precedents

**Channel Pruning, ICCV 2017:** alternates channel selection with least-squares reconstruction of the following layer's output. Its multi-layer treatment accounts for upstream approximation errors. Its residual-network treatment is particularly relevant to preserving a complete residual output rather than matching a branch while ignoring its changed skip input. The authors provide an implementation. This supports changing the retained weights after selection; it does not establish our selected channels are sufficient for speech. [Paper](https://arxiv.org/pdf/1707.06168), [author repository](https://github.com/ethanhe42/channel-pruning)

**Asymmetric reconstruction:** when approximating successive layers, uses the already-compressed network's input to predict the original network's target output. This directly supports recomputing actual student inputs after each earlier fit, instead of fitting every operation independently on pristine teacher inputs. Its results concern image networks. [Paper, section 3.3](https://arxiv.org/pdf/1505.06798)

**ThiNet:** selects channels using their effect on the next layer, applies a reconstruction-based rescaling initialization, and fine-tunes between pruning steps. This is closer to the user's proposed gradual recovery than merely slowing the optimizer. The project and code are available; its image-network accuracy and speed numbers cannot be transferred to AudioVAE2. [Paper](https://arxiv.org/pdf/1707.06342), [author project and implementation](https://www.lamda.nju.edu.cn/luojh/project/ThiNet_ICCV17/ThiNet_ICCV17.html)

**Once-for-All:** progressively introduces smaller subnetworks while retaining larger ones in shared-weight training. It supplies a broader precedent for gradual shrinking with distillation, but building that supernetwork would be substantially more machinery than our present fixed-width recovery needs. [Paper](https://arxiv.org/abs/1908.09791), [author implementation](https://github.com/mit-han-lab/once-for-all)

**A gentle transition is different from a smaller learning rate.** Asymptotic Soft Filter Pruning increases the number of pruned filters gradually and allows pruned filters to update during training. BERT-of-Theseus instead progressively raises the probability of replacing original modules with compact substitutes. These are vision and language methods respectively, not demonstrations of causal codec quality. [ASFP paper](https://arxiv.org/abs/1808.07471), [SFP author implementation](https://github.com/he-y/soft-filter-pruning), [BERT-of-Theseus paper](https://aclanthology.org/2020.emnlp-main.633/)

A future audio adaptation could reduce teacher assistance gradually at the shared whole-group boundary. That is safer than repeatedly inserting teacher-coordinate terms inside an already adapted student. Such assistance can hide student errors, so the pure student must be evaluated throughout, and assistance must reach zero before acceptance. This is a conditional proposal, not an additional arm in the approved calibration experiment or a technique demonstrated by StreamCodec2.

## Application to our measured failure

The new diagnostic is more specific than a general distillation argument. At pristine 384/256 initialization, restoring the omitted stage 3 upsampler contribution makes all 13 startup windows pass. Individual residual-unit restorations worsen aggregate startup error, while restoring all four sites recovers the teacher within the accepted interface tolerance. The measured signed cross terms are positive, so this is not evidence for the earlier negative-cancellation hypothesis.

For the ten exactly zero-input starts, the recovered model produces an almost identical residual pattern across sources. This supports a fixed startup-response defect, not language-specific noise. It does not prove a fitted native upsampler can synthesize the missing response from its retained inputs. The oracle uses teacher features absent from the deployed student.

The approved two-arm initialization experiment is appropriately bounded:

1. Reconstruct only the native stage 3 upsampler using its actual fresh-slice input.
2. Reconstruct the three residual pointwise mixers in order, recomputing inputs each time, then reconstruct the upsampler.

For a residual mixer, the fitting target must be the teacher's complete selected residual-unit output minus the current student skip. For the upsampler, retain native stride 5/kernel 10 and one shared output bias across phases. These are proposed causal-audio adaptations of reconstruction methods, not a reproduced StreamCodec2 recipe.

Use the fixed 72 calibration sources only for fitting and the fixed 96 development sources for before/after evaluation. Preserve true startup geometry and partial-cell weights. Report startup and ordinary speech separately; a local feature improvement can still damage the waveform. The calibration set contains 29 true-source starts, but their first 20 ms occupy only 0.337% of valid samples, and not all starts are silence.

Keep the 5,000-step checkpoint unchanged. Teacher-coordinate fitted weights should not be inserted independently into its coadapted internal representation. If either fresh initialization is promising, the next decision is bounded joint recovery against the original full-group and waveform teacher targets. Subsequent width cuts should follow measured recovery and review, not an assumed number of slow updates. No new inference layers or arithmetic are needed for these fitted-weight variants; CPU speed remains to be measured rather than inferred from a vision paper.
