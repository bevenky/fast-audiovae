# Original-history GPU decoder: static shape audit

This is a read-only audit of `outputs/apple-gpu-v2/mps_decoder_before.py` and the matching cached 40/80 ms generated wrappers. No model execution or new timing was performed. [shape-audit.json](shape-audit.json) records all 45 operations, literal weight names, input/output shapes, wrapper lines and hashes. All 90 convolution shape/attribute checks passed.

Each packet is B=1, FP32, with 1/2 AudioVAE2 latent frames and 1,920/3,840 output samples. The original-history model retains 26 tensors totaling 647,424 bytes. It is distinct from the projected-overlap candidate.

| Region | Transpose Cin→Cout; kernel/stride | Transpose new T, 40/80 ms | Three residual blocks: C; T, 40/80 ms | Region MACs, 40/80 ms |
|---|---|---|---|---|
| Stem | DW64, K7; pointwise64→2048 | 1 / 2 | None | 0.132M / 0.263M |
| Stage 1 | 2048→1024; K16/s8 | 1 / 2 | 1024; 8 / 16 | 92.447M / 151.339M |
| Stage 2 | 1024→512; K12/s6 | 8 / 16 | 512; 48 / 96 | 94.888M / 183.484M |
| Stage 3 | 512→256; K10/s5 | 48 / 96 | 256; 240 / 480 | 112.701M / 224.092M |
| Stage 4 | 256→128; K4/s2 | 240 / 480 | 128; 480 / 960 | 56.472M / 112.812M |
| Stage 5 | 128→64; K4/s2 | 480 / 960 | 64; 960 / 1920 | 28.848M / 57.663M |
| Stage 6 | 64→32; K4/s2 | 960 / 1920 | 32; 1920 / 3840 | 15.061M / 30.114M |
| Final | Conv32→1, K7 | 1920 / 3840 | None | 0.430M / 0.860M |

Each residual block has a K7 depthwise convolution, with dilation 1, 3 or 9, followed by a C→C pointwise convolution. Its depthwise input includes 6d prior samples. Each original transpose convolves T+1 inputs, including one saved activated input, produces `(T+2)s` raw samples, and emits only Ts. These details follow source lines 65–106 and 235–260.

| Family | Operations | MACs, 40 ms | MACs, 80 ms |
|---|---:|---:|---:|
| Transpose convolution | 6 | 243,179,520 | 445,030,400 |
| Pointwise convolution | 19 | 151,519,232 | 303,038,464 |
| Depthwise convolution | 19 | 5,849,536 | 11,699,072 |
| Final K7 convolution | 1 | 430,080 | 860,160 |
| Total | 45 | 400,978,368 | 760,628,096 |

These are logical tap multiply-accumulates; two FLOPs per MAC. They exclude bias, nonlinearities, copies and backend-specific algorithms. Their percentages are not runtime shares. The 43 Snake operations process 1,951,744/3,903,488 channel-time elements, with three multiplies, one sine and one add per element. Six affine pairs and 18 residual adds are separate source expressions, but compilation fuses many of them.

The cached original-history wrappers have 45 `extern_kernels.convolution` calls with `bias=None`, plus 83/84 generated Metal calls using 70/71 distinct generated kernels. Bias, Snake, residual addition and some concatenation work already share Metal kernels. There is no explicit matrix multiply or activation-axis transpose in these call bodies; `reinterpret_tensor` creates views. This does not establish the convolution backend's internal layout or algorithm.

Explicit eager source concatenations write destination buffers totaling 5,112,320/9,577,216 bytes per packet. State clones total 647,424 bytes, and the six transpose crop copies total 1,114,112/2,228,224 bytes. These are source allocation/copy sizes, not compiled traffic. The compiled wrappers instead contain 59/61 explicit tensor allocations and 17/18 reinterpret views, with reuse and fusion. Their allocation capacities sum to 7,610,624/14,407,680 bytes, excluding convolution internals; this is neither peak memory nor measured bandwidth. All 27 external history/input arguments have conditional alignment-copy guards, which do not prove copies occurred. The 40 ms wrapper returns its first transpose state as a strided `[1,2048,1]` view, causing a further 8,192-byte contiguous materialization in `experiment.CompiledModel.decode`; all 80 ms returned states are already contiguous.

## Equivalent alternatives worth isolating

1. **Pointwise, weight-left GEMM.** For all 19 pointwise nodes, `W[:,:,0]` is a contiguous `[Cout,Cin]` view and `x[0]` is `[Cin,T]`. `torch.mm(W[:,:,0], x[0])` followed by one bias addition and a batch-axis view produces contiguous BCT output without activation transpose or temporal window materialization. There are seven geometries: stem64→2048 at T1/2 and six square residual geometries in the table, three nodes each. The C256/T240–480 and C512/T48–96 groups have the largest pointwise arithmetic totals; select concrete targets using the profile. A channels-last `F.linear` route is also equivalent, but its output transpose may need materialization before the next BCT convolution, so that cost belongs inside a comparison.

2. **Transpose as two phase GEMMs, retaining input history.** Prepack constant weight halves into `[Cout*s,Cin]` matrices. Apply the first half to new x, and the second half to `[history,x[:-1]]`, each with T columns. Reshape each result into channel/time/phase order, add current and previous contributions, then add bias once. This preserves original input-history semantics while omitting discarded edge phases. Across six stages it removes 41,328,640 logical MACs per packet. It also introduces phase assembly/layout work and may change reduction order; no speed gain follows from the arithmetic count. The already-tested projected-overlap formulation did not establish an added gain beyond compilation, so this is a distinct GEMM mapping, not evidence to repeat that claim.

The final K7 convolution could use an unfolded `[224,T]` input and a matrix multiply, but materializing those windows for only 0.11% of logical MACs is a weaker static candidate. Depthwise has only seven multiplies per output element, making a dense GEMM mapping unattractive without unnecessary gather or zero work. Every alternative still needs the original FP32 waveform/state tolerances and complete input/output layout costs.
