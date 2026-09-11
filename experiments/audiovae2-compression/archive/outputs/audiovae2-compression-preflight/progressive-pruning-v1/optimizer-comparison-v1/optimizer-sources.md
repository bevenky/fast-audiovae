# Four optimizer recovery comparison

This is a controlled implementation of matrix optimizer formulas for the existing weight-normalized decoder. It is not a claim of bitwise equivalence to a released optimizer package or of faster codec recovery. No upstream package is installed, and no inference operation changes. `recovery_optimizers.py` is the implementation; each launch must record its byte hash and `optimizer_config()` snapshot.

## Fixed parameter boundary

Exactly nine native parameters use a matrix method: `model.{3,4,5}.block.{2,3,4}.block.3.weight_v`. They are the three residual pointwise convolutions at each trainable stage, with square widths 384, 256 and 128. Each `[output,input,1]` tensor is independently viewed as `[output,input]`. Layers are never stacked, concatenated or flattened together. NorMuon averages across the input columns of this 2D view, so its extra state is `[output,1]`.

All other 81 trainable tensors retain real PyTorch AdamW with learning rate `3e-5`, betas `(0.9,0.99)`, epsilon `1e-8` and weight decay zero. This includes weight-normalization gains, biases, Snake parameters, depthwise kernels and transposed convolutions. All original 90 parameter objects and their canonical ordering remain authenticated even though a mixed optimizer has separate 81/9 physical groups.

The AdamW control is the actual `torch.optim.AdamW` class. At a pointwise learning rate of `3e-5` it uses the original single 90-parameter group. At another qualification learning rate only the nine pointwise parameters move to a second AdamW group; the other 81 retain their original rate. The source creates no substitute Adam update for the control or the 81-parameter complement.

## Primary references and immutable pins

| Reference | Pin and role |
|---|---|
| [Author Muon implementation](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py#L3-L37) | Commit `f98f1cacc0263b04290753e32be8d498c1efc806`; Newton-Schulz polynomial, optional Nesterov and momentum update. |
| [NorMuon paper, Algorithm 1](https://arxiv.org/html/2510.05491v1#S3) | Immutable arXiv `2510.05491v1`; neuron-row second moment and final RMS scaling. |
| [Author NorMuon implementation](https://github.com/zichongli5/NorMuon/blob/c6989a8354730695d9f5a9faa6c55eeb24865209/normuon.py#L29-L50) | Commit `c6989a8354730695d9f5a9faa6c55eeb24865209`; reference for differences between paper and released defaults. |
| [Original Shampoo paper](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf) | ICML 2018, Algorithm 2; separate left/right factors and inverse fourth roots for a matrix. |
| [Distributed Shampoo paper](https://arxiv.org/pdf/2309.06497v1) | Immutable arXiv `2309.06497v1`; practical grafting, amortized preconditioners and CNN evidence. |
| [Official Distributed Shampoo implementation](https://github.com/facebookresearch/optimizers/blob/f18f735c972d304542af15e62b5acaa503169f2b/distributed_shampoo/distributed_shampoo.py#L152-L184) | Commit `f18f735c972d304542af15e62b5acaa503169f2b`; diagonal grafting with a shared filtered gradient and optional blocking. |

The pins identify the source versions inspected, not a claim that they are the latest releases. The source files at all three code pins were read directly before implementing this comparison.

## Declared formulas

Let `G` be one native pointwise `weight_v` gradient viewed as a matrix. Buffers start at zero. `RMS(U)=||U||F/sqrt(m*n)`. Every update below is applied as `v <- v - matrix_lr * U`; no post-update weight normalization or reparameterization is performed by the optimizer.

### Muon and NorMuon

Both arms use momentum `M <- 0.95*M + 0.05*G`, without Nesterov. The shared orthogonalizer uses five FP32 iterations, starting from `X=M/(||M||F+1e-7)`. If rows exceed columns it transposes for the calculation, then restores the original axes. Each iteration is `X <- a*X + (b*A+c*A*A)*X`, where `A=X*X.T` and `(a,b,c)=(3.4445,-4.7750,2.0315)`.

Muon rescales the resulting direction to RMS `0.2`. NorMuon first computes `s <- 0.95*s + 0.05*mean_columns(X*X)`, divides each row by `sqrt(s)+1e-8`, then also rescales to RMS `0.2`. The row second moment has no bias correction. A zero direction remains zero without division by zero. This follows NorMuon's paper-level normalization and makes the two arms differ only in neuron adaptation. [NorMuon Algorithm 1](https://arxiv.org/html/2510.05491v1#S3)

These are explicit experimental choices: current author code defaults to Nesterov and BF16 Newton-Schulz; the Muon code uses aspect-ratio scaling, while the NorMuon code preserves the pre-normalization norm before that scaling. Here both use the supported non-Nesterov momentum policy, FP32 arithmetic, and exact common RMS `0.2`. Epsilon is fixed as stated above. Therefore these arms should be described as the declared Muon/NorMuon variants, not exact default-library reproductions. [Muon source](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py#L3-L37), [NorMuon source](https://github.com/zichongli5/NorMuon/blob/c6989a8354730695d9f5a9faa6c55eeb24865209/normuon.py#L29-L50)

### Shampoo with Adam-style norm grafting

This arm uses independent unblocked left/right factors per pointwise matrix. All matrix statistics, inverse roots and intermediate direction arithmetic use FP64; the final native update is cast to FP32. It keeps:

```
M <- 0.9*M + 0.1*G
L <- 0.99*L + 0.01*G*G.T
R <- 0.99*R + 0.01*G.T*G
V <- 0.99*V + 0.01*(G*G)
```

Each EMA is divided by its corresponding `1-beta**step`. With hats denoting those corrected values, `A=Mhat/(sqrt(Vhat)+1e-8)`. At steps 10, 20, 30 and so on, compute `PL=(Lhat+1e-12*I)^(-1/4)` and `PR=(Rhat+1e-12*I)^(-1/4)` using a symmetric FP64 eigendecomposition. Reuse those roots between refreshes. Steps 1 through 9 use `A`; later steps use `S=PL*Mhat*PR` followed by `U=S*||A||F/||S||F`. No extra post-grafting momentum is applied. A nonzero graft with zero Shampoo direction or a nonpositive/nonfinite regularized eigenvalue is a hard error, with no adaptive jitter or fallback.

The two-factor exponent comes from matrix Shampoo. The graft uses the same filtered gradient in the diagonal and matrix directions, as the official implementation describes. Our exact EMA rates, fixed absolute damping, FP64 choice, ten-step refresh and absence of blocking are declared comparison settings. They are not copied performance guarantees. [Original Shampoo Algorithm 2](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf), [official grafting description](https://github.com/facebookresearch/optimizers/blob/f18f735c972d304542af15e62b5acaa503169f2b/distributed_shampoo/distributed_shampoo.py#L152-L184)

## Qualification settings and interpretation

The approved qualification has three learning rates per method, each starting afresh for 64 updates. AdamW and Shampoo use pointwise rates `{1.5e-5,3e-5,6e-5}`; Muon and NorMuon use `{7.5e-5,1.5e-4,3e-4}` with RMS `0.2`. These grids are experimental choices. Select rates using the fixed 12 non-anchor calibration sources and the existing objective, without development selection. Initialization, ordinary source order, loss coefficients, accumulation and startup policy must otherwise match. Repeated six-source calibration anchor exposure remains separate from the unique ordinary-source ledger.

The matrix methods operate on weight-normalization direction coordinates. For one row, `v=r*u` and `W=g*u`, so to first order `dW = u*dg + (g/r)*(I-u*u.T)*dv`. Consequently orthogonalizing or preconditioning `dv` is not the same as doing so to effective `dW`. Gains still use AdamW, and the final startup correction can change the proposal's matrix geometry. Neither interaction is automatically a bug or a convergence guarantee.

Shampoo has published CNN experiments, including ResNet/ImageNet in the distributed implementation paper. Those results motivate testing it, but they do not establish its value for nine square, weight-normalized codec matrices or for our final waveform feasibility correction. [Distributed Shampoo experiments](https://arxiv.org/pdf/2309.06497v1)

With the present widths, Shampoo holds six FP64 arrays per square matrix after refresh: 4,128,768 values, approximately 31.5 MiB for the nine matrix states before serialization overhead. A root refresh requires 18 eigendecompositions and is scheduled every ten updates. This is an analytic storage/work count, not a measured H100 overhead. Wall time and accepted displacement must accompany loss comparisons.

## State and transaction contract

`optimizer_config()` returns a deep-copy JSON-compatible snapshot. Live group membership, canonical 90 ordering, matrix identity, shapes and hyperparameters are checked against the factory seal. At update zero state must truly be empty. Thereafter every native parameter has exactly one real monotonically advancing counter. Matrix parameters store their method's buffers rather than fabricated Adam moments.

The mixed optimizer delegates its 81-parameter complement to actual PyTorch AdamW. Its state dictionary includes the configuration; reload rejects a different recipe and explicitly restores Shampoo's serialized FP64 values without the default PyTorch FP32 cast. The inner Adam object is rebound to the restored shared state. The retention wrapper owns full parameter, optimizer and RNG rollback on hard failure.

Startup constraints are evaluated on the actual full 90-parameter proposal after its one optimizer step. The unchanged nonlinear waveform gate determines acceptance; it is not replaced by any optimizer statistic. Accepted zero parameter movement still advances the selected optimizer's genuine state once, as in the existing declared retention policy. Any such events must be reported. No optimizer states or updates are silently synthesized to satisfy the ledger.
