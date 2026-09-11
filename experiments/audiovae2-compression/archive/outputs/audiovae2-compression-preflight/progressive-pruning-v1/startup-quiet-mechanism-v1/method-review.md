# Current-cut silence diagnosis: method review

This is a read-only diagnostic of the current **384/256** student. Only the first internal width changed from 512 to 384; all nine residual units, strides, dilations and causal history remain. The older256/128 restoration experiment does not supply causal measurements for this cut.

## Fixed panel

`panel.json` contains 165 exact 20 ms windows from 19 of the fixed 96 development sources. Its SHA256 is `4a393e34913d0c6d5452db59a7aa4eff279d1a637202064354738a632652ac9f`.

- All 13 teacher-near windows at original source 0–20 ms. Ten have exactly zero reference input; three do not.
- All 114 teacher-quiet windows from the predeclared Punjabi, Bodo, Somali, whistle 428921 and Thorsten whisper recordings, including any passing windows.
- Ten paired source-zero 20–40 ms teacher-transient windows.
- Kannada source 40–60 ms and 800–820 ms controls, and a contiguous 160 ms Spanish interior control represented by eight original20ms windows.
- The earliest complete teacher-active 20 ms window after 40 ms for each selected source where one exists. These are same-source active controls, not amplitude-matched examples. Luxembourg has no such scored active window.

Selection uses only immutable teacher/window metadata and the original crop manifest. The teacher-window identity `b74504a0ed422cc106510873b9a270da0d32c4b4b8dd34bd9f31858fb0753660` matches the current panel and identical-teacher control. No older student's outcomes were used. The diagnostic must authenticate source membership, crop coordinates, this full identity, and runtime teacher classifications before interpreting any selected window.

## What the existing control clears

The original-weight independent copy matches native teacher execution on all 60,000 scanned crops. All 1,153,650 quiet windows and 58,045 near-silence windows pass. The first 144 fitting sources also have zero loss and gradient during 12 optimizer steps, with unchanged weights. This rejects a systematic evaluator, copy-wrapper or mask defect as the explanation for the known 13 startup failures.

It does not prove every cached waveform is bitwise identical: one source has a within-tolerance maximum difference2.53e-7, outside those 144 optimizer examples. One cold shape also showed transient rounding. Neither observation explains the known startup failures, which reproduce against exact same-input native targets.

The first-cut step 0 already fails all 2544 quiet windows. Recovery to 5000 reduces failures to 1033; all 13 startup windows remain included in that total. Current startup residual RMS 22.004 microFS versus centered RMS 21.746 microFS indicates predominantly time-varying error. A uniform offset correction alone is therefore unsupported. Sustained source-zero 164/164 and interior-near 50/50 currently pass, although those upper-bound checks do not require an exact teacher noise floor.

## Four-site accounting and basis safety

The direct lost-input sites are the three stage 2 residual pointwise mixers and the stage 3 upsampler. Paths are `model.3.block.{2,3,4}.block.3` and `model.4.block.1`. Stage3 upsampling is stride 5/kernel 10, with right trim 5. Stage4's width is unchanged in this cut.

At pristine sliced step 0, a complete residual-unit gap separates into retained skip drift, retained-input mixing drift, and the teacher's dropped-input contribution. Keep any measured floating-point reconstruction remainder separate. For cancellation, the retained component must include both the original bias and residual skip. Report signed cross terms, not just contribution magnitudes.

At stage 3, split dropped-input and retained-input-drift responses, and distinguish current and previous input-frame contributions for each of five phases. The previous-frame contribution is absent at startup. Restoring all four teacher-derived missing terms is an accounting control at step 0, not a deployable repair. Partial restorations may interact through downstream nonlinearities and need not improve monotonically.

Do not inject teacher-coordinate terms into the adapted 5000 representation. Observe its internal features descriptively; use the shared full group-output and waveform boundaries for teacher substitutions or quantitative reconstruction claims. A fresh-slice result cannot uniquely attribute the remaining trained-model error.

## Why startup deserves a separate explanation

The original source uses left zero padding for causal convolution, native transposed convolution with right trimming, channelwise Snake, learned convolution biases, and sample-rate scale/bias conditioning. These are preserved. See `audio_vae_v2.py:20–38,50–65,75–99,176–210,259–267,310–354` under `work/convnext-natural-context-repair/upstream-source-audit/`.

Zero input audio does not imply zero encoder latents or zero decoder features. Snake itself maps zero to zero, but its inputs can be nonzero after learned conditioning and bias. Missing causal history at startup can therefore change the contribution balance without any padding implementation error. A negative kept/dropped cross term together with a controlled restoration would establish cancellation at that measured site; it would not establish that Snake, bias or padding is independently faulty.

The selector uses an uncentered stage-end activation Gram with at most 256 randomly chosen valid time cells per calibration source, then pivoted channel selection (`prepare_progressive_schedule.py:29–55`). The initializer slices effective weights (`group_model.py:227–272`). Selection does not optimize weighted downstream reconstruction or startup error, and slicing does not fold predictable removed-channel contributions into retained weights. This makes lost cancellation a concrete hypothesis even when channel features are redundant, not proof that remaining capacity is inadequate. Stage-end redundancy also does not establish predictability at earlier post-Snake residual inputs or at the conditioned post-Snake upsampler input. Any future folded correction must be assessed at its actual operation input.

The 60,000-source exposure audit finds 15.6736% quiet samples and 29,196 true-source-start crops. First 20 ms samples across all levels occupy 0.397073% of waveform exposure. This rules out universally masked starts and broadly absent quiet data. It does not measure silent-start gradient mass or prove the current objective sufficiently prioritizes this error.

## Interpretation guardrails

Compute DC per channel over valid time, then aggregate energies; averaging channels first can hide offsets. Subtract in FP64 before DC/AC statistics. Map waveform validity into hidden-rate cells with partial-cell sample weights. Estimate periodicity only within contiguous spans with enough cycles; never concatenate unrelated quiet windows. Repetition alone is not proof of aliasing. A 20 ms startup window cannot support a long-period spectral claim.

The useful outcome is localization of immediate slicing loss, its propagation, and what recovery has or has not repaired. Neither successful oracle restoration nor failed recovery proves an irreducible width limit. No new loss, architecture, pruning cut or training run is proposed by this audit.
