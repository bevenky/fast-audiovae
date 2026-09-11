# Distillation methods relevant to this decoder

The strongest lesson is to distinguish learning useful intermediate features from reproducing the teacher's actual decoder function. Our frozen encoder already supplies the same 64-channel latent tensor to both decoders. There is no separate student latent representation to align unless we change the encoder. The task is to make the smaller group transform those same inputs into features the original frozen suffix can use correctly.

## Three primary findings

**FitNets: feature hints can help a thinner student, but are not the final task.** Sections 2.2–2.3 introduce a regressor between differently sized hidden representations, pretrain the guided student portion, then train the whole student using output distillation. The paper explicitly cautions that strong or deep hints can over-regularize the student. Its results are vision benchmarks, not audio reconstruction. This supports auxiliary feature guidance and subsequent joint optimization, not independently repairing every layer and assuming the assembled decoder is correct. The auxiliary regressor is outside the deployed task path. [Romero et al., FitNets, ICLR 2015](https://arxiv.org/html/1412.6550v4).

**Teacher Assistant KD: a larger teacher is not automatically an easier teacher.** The paper observes weaker transfer across some large teacher/student capacity gaps, and improves classification by first distilling an intermediate-sized assistant. Its explanation partly concerns softened class logits, which our deterministic waveform target does not have. It is therefore a precedent for staged compression, not evidence that our current width reduction has reached an irreducible limit. An assistant adds training work; the final deployed student need not become larger. The paper does not establish an optimal audio-decoder assistant width or guarantee faster convergence. [Mirzadeh et al., Improved Knowledge Distillation via Teacher Assistant, AAAI 2020](https://arxiv.org/html/1902.03393v2).

**Relational KD: matching geometry allows different representations, but cannot replace exact reconstruction.** RKD transfers normalized pairwise distances and triplet angles instead of matching individual feature vectors. This accommodates different embedding dimensions. Crucially, section 3.2.4 says relations alone are inadequate when individual output values matter. Its experiments concern metric learning, classification and few-shot learning. For our audio task, a relation loss could provide an auxiliary structural signal, but it must not replace sample-aligned waveform and full-boundary supervision. Normalized distances and angles do not fix absolute amplitude, offsets or the feature coordinate system consumed by the frozen suffix. [Park et al., Relational Knowledge Distillation, CVPR 2019](https://arxiv.org/html/1904.05068v1).

## How this maps to the current pruned AudioVAE2

| Supervision | Meaning in our decoder | Main limitation |
|---|---|---|
| Outputs | Same latent/context input; compare actual decoded waveform and its spectrum with the frozen teacher | A good spectral or correlation score alone can coexist with amplitude error |
| Features | Match the complete 128-channel group output in the original teacher coordinates | Internal narrower representations need not preserve the teacher's individual channels |
| Relations | Preserve selected relationships between aligned feature frames or examples | Can leave the absolute scale, offset or basis unconstrained |

The unprojected 128-channel boundary loss already exists in our recipe. FitNets does not reveal a missing generic feature-loss switch. Adding more hints should address a diagnosed difficulty, rather than duplicating the existing objective or constraining every intermediate layer.

A learned projector creates a specific measurement trap. If `P(h_student)` matches `h_teacher`, this does not prove that the frozen suffix receives the right `h_student`. For example, the projector can undo a scale or basis change that the suffix itself cannot undo. That is a mathematical identifiability limitation, not evidence that a projector will necessarily fail. At the shared final boundary, retain direct matching without a learned escape route. If an internal-width projector is tested, keep it a small training-only auxiliary head and judge success through the actual decoder without it.

Likewise, a large feature-loss coefficient can spend gradient effort reproducing a representation instead of improving the waveform. Our measured first-mixer fit already illustrates the distinction: its heldout local SSE improves 75.86%, while full-decoder waveform quality does not. That result supports requiring both local and downstream evidence before retaining an auxiliary method.

## Recommendation

Preserve the original frozen encoder and teacher, the actual student chain, and direct full-boundary plus waveform supervision. The most relevant literature-backed direction is coordinated group distillation with carefully limited hints, rather than isolated layer repairs or relation-only matching. Hints should use the student's actual upstream activations; teacher-forced inputs are useful diagnostics but can hide propagated student error.

Treat an intermediate-width assistant as a fallback if evidence shows the compression jump is difficult to learn. It must be judged against the original teacher and the same quiet, expressive and waveform checks. Do not introduce an assistant merely because it is described in the literature, or replace the original quality reference with a weaker target.

Training-only hints, projectors and assistants can leave inference architecture and CPU work unchanged when excluded from the deployed path. None of these papers demonstrates AudioVAE2-quality reconstruction, unchanged streaming behavior, or a particular CPU RTF for this model. They justify methods to investigate, not predicted results.

No code, model, training or benchmark was executed for this literature review. StreamCodec2 is being reviewed separately.
