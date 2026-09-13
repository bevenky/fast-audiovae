# Apple GPU V3: short diagnostic and matrix screens

September 13, 2026. These are short Apple M5 Max MPS FP32 tests with PyTorch 2.14.0, one CPU caller, CPU fallback and fast math disabled. They do not establish exclusive GPU layer durations, sustained latency, or broad perceptual quality. No production default or trained tensor changed.

The combined matrix candidate cut completed-output time by 51.25% at 40 ms and 51.50% at 80 ms in the short back-to-back screen. A separate 80 ms paced probe retained a 32.06% mean service-time reduction. Both the combined and transpose-only candidates failed the unchanged internal-state numerical gate, so neither is promoted. Waveform comparisons passed up to the qualification failures and throughout the short timing screens.

## Host, completion and queue measurements

[Host profile analysis](profile-analysis.md) independently reconciles 54 profiled/unprofiled packets and all component sums. With the hooks disabled, mean completed-output times were **4.813 ms at 40 ms** and **5.064 ms at 80 ms**. Model host submission spans were 1.692/1.743 ms; the combined finite-check-and-wait spans were 2.696/2.902 ms. The latter includes unfinished model work and cannot be called validation overhead alone.

Each instrumented packet contained **19 depthwise convolutions, one final ordinary convolution, 19 pointwise convolutions and six transpose convolutions**, plus 83/84 generated Metal dispatches. Hook spans include CPU submission, driver work and any blocking within each call. Generated kernels already combine bias, Snake, residual and state work. The unhooked remainder also contains instrumentation bookkeeping; it is not an isolated Python cost. Off/on/off timings varied enough that this run does not isolate instrumentation overhead from background variation.

[Queue probe](queue-probe-r1.json) measured public steady means of **6.749 ms unpaced** and **12.160 ms with 80 ms pacing**. In its split-barrier diagnostic, completion waits averaged 3.716/4.774 ms and subsequent finite checks 0.599/0.514 ms. The added barrier changes execution, so those spans do not establish precisely removable overhead. Every public call completed GPU work and copied its audio before returning. **No prior-packet queue backlog was demonstrated.** Slower paced service alone does not identify power, clock or scheduler causes.

[QoS probe](queue-qos-r1.json) began and ended at class **33, USER_INTERACTIVE**. The trial class **25, USER_INITIATED**, is lower priority. This was not a priority increase. Paced steady means were 10.050 ms at the initial class and 9.867 ms at the trial class, with substantial variation; there is no convincing QoS improvement. Across the short paced observations, no 80 ms deadline was missed and the largest ready-after-arrival delay was 21.772 ms. This is not sustained deadline qualification.

The first host-profile attempt failed on an unavailable `PyCodeCache.cache` attribute; [its log](profile-host-r1.log) is retained. The [Instruments capture](metal-r1.log) did not yield usable GPU attribution. The retained trace must not be presented as a successful GPU kernel profile.

RTF below one means service capacity exceeds the audio arrival rate. It does not mean zero latency. For example, an 80 ms packet at RTF 0.065 needs about 5.2 ms to decode. The public API waits for that packet's GPU work before returning CPU-ready audio, so this serial protocol does not accumulate unfinished earlier packets. Host dispatch, current GPU execution, synchronization and result handling remain real costs. Sleep-based pacing also introduces host arrival jitter. A read-only desktop snapshot showed substantial activity in other applications and services; the experiment did not stop or modify them. That is a source of possible interference, not proof of the specific cause of paced slowdown.

## Pointwise matrix screens

Each screen used the same three speech crops of 960 ms, fresh streams, 40/80 ms packets, one qualification sweep, one warmup sweep and two measured sweeps. Order reversed within each clip/packet group across the two measured repetitions. All arms used the same public GPU input/output and all-history validation boundary. Compilation was excluded; ordinary lazy state initialization and completed output remained timed.

| Matrix mode | Arm | RTF 40 ms | RTF 80 ms | Median paired reduction, 40/80 ms |
|---|---|---:|---:|---:|
| Default | Original compiled | 0.126139 | 0.067629 | Reference |
| Default | Compiler 1×1-as-MM option | 0.124853 | 0.067326 | 1.22% / 1.01% |
| Default | Explicit weight-left pointwise MM | 0.120057 | 0.064370 | 4.96% / 5.00% |
| Prefer Metal | Original compiled | 0.162658 | 0.091198 | Reference |
| Prefer Metal | Compiler 1×1-as-MM option | 0.166365 | 0.098682 | −2.11% / −8.18% |
| Prefer Metal | Explicit weight-left pointwise MM | 0.162935 | 0.092303 | 1.17% / −1.90% |

RTF is pooled packet-plus-flush time divided by emitted duration. Paired reductions use six matched clip/repetition pairs. Explicit pointwise MM won all six pairs at both packet sizes in the [default-dispatch run](matmul-default-r1.json). It did not give a consistent gain in the [Prefer-Metal run](matmul-metal-r1.json). The controls themselves differed materially across those separate invocations; do not attribute that entire cross-run difference to the environment flag. The within-run controls are the relevant comparisons.

Both runs passed all 72 stream comparisons and six preparation comparisons, with worst stream waveform difference 3.502e-7 against the original CPU ONNX reference at the unchanged 1e-5/1e-4 tolerances. Compiler counters stayed fixed after six prepared graphs.

## Logical shapes versus measured cost

[The shape audit](shape-audit.md) matched all 45 original-history convolution shapes and attributes to both cached wrappers. At 40/80 ms, logical MACs are **400.978M/760.628M**: transpose 243.180M/445.030M, pointwise 151.519M/303.038M, depthwise 5.850M/11.699M, and final convolution 0.430M/0.860M. Those proportions describe arithmetic, not runtime.

The 19 pointwise convolutions admit contiguous `[Cout,Cin] @ [Cin,T]` multiplication without a temporal window or activation transpose. The six transpose convolutions admit a packed phase projection plus overlap assembly. This can change dispatch and layout costs even when the underlying logical multiplication count is unchanged. Source-level copy totals also differ from compiled traffic because concatenation, bias and state copies are fused or reused. Backend-internal scratch, layout and GPU occupancy remain unmeasured here.

## Transpose screen: large gain, qualification failure retained

The [transpose-r1 receipt](transpose-r1.json) uses the same matched short protocol. The performance measurements are complete; the separate quality qualification below prevents promotion.

| Arm, default dispatch | RTF 40 ms | RTF 80 ms | Median paired reduction, 40/80 ms |
|---|---:|---:|---:|
| Original compiled | 0.122989 | 0.065275 | Reference |
| Six transpose operations as packed MM | 0.066826 | 0.032925 | 44.96% / 50.02% |
| Transpose MM plus 19 pointwise MM | 0.059956 | 0.031658 | 51.70% / 50.74% |

Both alternatives won all six pairs at each packet size. All 72 streamed waveform comparisons and six preparation comparisons passed; worst stream difference was 3.604e-7. Six compile graphs were prepared and counters stayed fixed during the sweeps. The preliminary CPU algebra gate passed 48 single-call cases plus 12 uneven-stream/future checks across strides 2, 5, 6 and 8.

Weights were permuted and packed once before compilation; original buffers were also retained. Packing duplicates 165,314,560 bytes of weights in this prototype. The operation preserves prior activated-input histories, emits exactly the original sample counts and adds bias once. For filter length twice the stride, each output phase is the sum of a current-input projection and a previous-input projection. This is the existing transposed convolution expressed through matrix multiplication, with no learned operation or lookahead added.

[Compiled-source inspection](compiled-matrix-audit.md) confirms 25 matrix calls and 20 remaining convolutions in each combined graph. All six transposed and 19 pointwise convolution calls are replaced. The 45 external mathematical operations and logical MAC count are unchanged; generated Metal call sites increase from 83/84 to 84/85. Thus fewer mathematical calls do not explain the gain. The result supports the more efficient matrix execution route for these shapes. The phase sum, bias and following activation are fused in generated Metal; backend-exclusive GPU attribution remains unavailable.

## Paced arrival check on the combined candidate

[The final paced probe](transpose-paced-r1.json) uses the same eight 80 ms English packets twice per route, with balanced ordering. Mean service times exclude each initial packet; maxima and deadline checks include it. Timing includes the public API's validation and CPU-ready output. Both routes use the same FP32 weights and compiler settings.

| Route | Back-to-back mean | Paced mean | Paced maximum service | Maximum ready-after-arrival |
|---|---:|---:|---:|---:|
| Original compiled | 6.445 ms | 9.058 ms | 12.156 ms | 21.728 ms |
| Combined matrix candidate | 3.610 ms | 6.154 ms | 12.576 ms | 16.960 ms |

The paced mean improves 32.06%, but worst service time does not improve in this tiny sample. All 32 paced calls met their 80 ms completion deadlines; no stronger sustained guarantee follows. The largest observed start lateness was about 10 ms for both routes. All 64 waveform comparisons against sample-aligned original CPU ONNX output passed, with maximum absolute difference 1.956e-8. Two prepared graphs stayed fixed during ordinary calls. This paced protocol differs from the 960 ms back-to-back screen, so the 51% result must not be advertised as a 51% live-arrival improvement.

## Numerical qualification and next decision

The combined candidate and a separate transpose-only candidate each failed at `stage1.residual2.history` on the English fixture, after 35 model/API calls. The unchanged tolerances are atol=1e-5 and rtol=1e-4. The gates stopped at the failure, so later expressive, synthetic silence, reset and stream-isolation cases were not completed for these new candidates. Previous candidates' passing results do not qualify this one.

The [aggregate-only diagnostic](state-error-aggregate-r1.json) repeated the transpose-only failing case: one of the first failing tensor's 27,648 elements exceeded tolerance, by a maximum of 9.782e-6. Its maximum absolute difference was 4.342e-5, RMS difference 1.659e-6, and largest error/tolerance ratio 1.291. No raw audio or latent/state values were exported. The waveform gate still passed. The combined run recorded a cumulative maximum of 4.064e-5 for the same state before failing.

That state is an internal activated-input history, not a waveform. For the failing two-latent call its 54 stored positions are entirely from the current packet. Reordered FP32 reductions can propagate through Snake and residual stages while leaving very small final waveform differences. This is a plausible numerical mechanism, not proof of the first diverging operation or a reason to relax the gate. The transpose-only failure shows that the pointwise change is not required to trigger it.

Retain the matrix candidate and all receipts, keep the current accepted route, and isolate the first numerical divergence before integration. A targeted fallback to the original operation at an affected stage is one option to test if accurate matrix accumulation cannot meet the existing gate. Do not start a longer corpus benchmark or remove validation to improve the timing number. CPU remains the default; GPU remains explicit opt-in.

## Primary implementation references

- [PyTorch MPS convolution implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/mps/operations/Convolution.mm): the backend route being replaced in the experiment.
- [PyTorch MPS matrix implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/mps/operations/LinearAlgebra.mm): matrix dispatch and the separate Prefer-Metal experiment.
- [PyTorch Inductor convolution lowering](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/_inductor/kernel/conv.py): the tested 1x1-as-matrix option.
- [Apple task priority guidance](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheTaskLevel.html): motivation for the bounded caller-QoS probe, not evidence that GPU frequency or priority was changed.
