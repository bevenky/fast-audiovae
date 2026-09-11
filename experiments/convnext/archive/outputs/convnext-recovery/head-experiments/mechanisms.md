# Why these experiments follow the diagnosis

The goal is to preserve the frozen AudioVAE2 encoder and the student's CPU-efficient ConvNeXt body. We first test whether its existing final synthesis matrix can learn the missing behavior before adding another layer.

| Case | Observed teacher behavior | Student evidence | Minimal experiment |
|---|---|---|---|
| Steady encoded silence | Internal phase variation is largely cancelled by the final learned multichannel synthesis convolution. Final tanh contributes essentially nothing at this amplitude. | Constant late features produce a repeating waveform through the 480-output readout. Most measured residual energy is shared across the four internal phases. | Fit the existing readout's stationary response, first the shared pattern and then all four phases. |
| Natural quiet audio | The output depends on changing latent content, including real low-level detail. | Most residual power changes over time. Subtracting a universal silence pattern explained little of the error. | Fit the existing readout to varied real teacher targets, with a separate quiet-audio validation gate. |
| Loud transients | The measured final-convolution peaks near 3 are compressed below full scale by tanh. | Our unbounded head overshoots. Reducing the last residual block's gain has conflicting local derivatives for peaks versus ordinary detail and quiet. | Fit the head to actual teacher pre-tanh values, then apply tanh. Compare with clipping only as a diagnostic control. |

The silence correction changes weights that act on every input. It is not a silence detector, output muting rule, or stored-waveform subtraction. That is also why it may fail: directions used for silence cancellation may carry useful breathing or quiet speech. Four constrained feature directions out of 2,048 is not proof of a small perceptual effect.

The real-audio fit is a defined convex optimization problem with the body fixed, a fixed regularization grid and source-disjoint selection data. Its failure would reject this fitting method, not establish that the architecture cannot represent the teacher. The pre-tanh fit similarly tests a migration method, not every possible bounded-head training recipe.

A result is useful only if the same change improves natural recordings excluded from this fit under the predeclared quality screens. Matching the synthetic silence fixture verifies the constraint, and clipping guarantees range; neither alone demonstrates better audio.
