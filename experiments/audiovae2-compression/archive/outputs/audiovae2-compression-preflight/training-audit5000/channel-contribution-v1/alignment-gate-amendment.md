# First-input numerical alignment check

The first diagnostic attempt stopped before affine fitting. Original teacher, initial student and protected files passed preservation checks. The complete full-width copied decoder matched the teacher bitwise on all 15 targeted recordings.

The failing calibration source was `fa_ir:train:11837461599893749470.wav` (index 2, no excluded context, 122,880 valid samples). Its narrowed first-mixer input differed from the selected teacher input by a maximum of `1.26362e-5` and RMS `2.36225e-7`; teacher input RMS was `0.2865004`. Relative RMS drift was approximately `8.25e-7`. The worst difference was at an activation of approximately `2.753337`, a relative error of `4.59e-6`.

The stage input conditioning and first Snake were bitwise equal. Differences began at the narrowed upsampler (maximum `8.5831e-6`, RMS `2.2139e-7`) and propagated through the following channelwise operations. Effective selected-weight differences were at most `1.49e-8` for the upsampler and `5.96e-8` for the depthwise filter. This supports floating-point rounding under changed convolution shapes and reconstructed weight normalization, rather than a wrong channel selection.

The fitting check compared the difference against zero with absolute tolerance `1e-5`. In contrast, the existing contribution and evaluation checks compare against the actual teacher signal with `atol=1e-5, rtol=1e-4`. The fitting check is amended to use that same scale-aware input comparison. Per-source maximum and RMS drift, signal RMS and the propagated retained-input contribution remain recorded. No teacher output, trained weight, fitting objective or quality threshold is changed.

The failed attempt and source snapshot remain preserved. A fresh output directory is required for the corrected diagnostic. First-input agreement is numerical, not bitwise; exact omitted-input terms and retained-input numerical drift remain distinct in the reported identity.
