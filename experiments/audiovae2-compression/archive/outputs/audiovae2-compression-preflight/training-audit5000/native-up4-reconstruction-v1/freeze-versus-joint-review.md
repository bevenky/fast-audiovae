# Freezing stages 2–3 versus joint recovery

The results do **not** justify freezing stages 2–3 as the next default recovery policy. Keep them trainable with stage 4 when continuing the connected decoder. Freezing the prefix can be useful for a short, explicitly isolated diagnostic, but the teacher-upsample substitution does not establish that joint adaptation was harmful.

## What supports joint training

Joint training improved the path feeding the upsampler, not only the final residual stack. With the same 72-source native-operator fitting method, development upsampler MSE falls from **0.00379348** using sliced-initialization inputs to **0.00149656** using the trained candidate's inputs. The latter is approximately **60.55% lower**. This comparison uses each model's actual propagated inputs and the same full teacher target. These inputs are captured after stage-4 conditioning and its input Snake, so the improvement cannot be assigned specifically to stages 2–3. It shows better linear predictability after joint learning across the upstream path.

The larger-accumulation comparison also recovered useful output behavior with all stages 2–4 trainable. From the common step-4500 reference, waveform MAE decreased from **0.00489207 to 0.00423436**, active correlation rose from **0.971874 to 0.976007**, and quiet residual RMS decreased. Changing the accumulation/update schedule altered the result substantially without freezing upstream layers.

The following matched baseline continuation improved MAE further to **0.00417039** and correlation to **0.976601**. This is evidence of continuing average progress, not proof that more training will reach the target. Near-silence failures worsened from **166 to 181 of 184** during that continuation, so progress is not uniform.

## What the native intervention actually proves

The trained stage-4 remainder has adapted to its current input. Feeding it exact teacher upsampler output produces active RMS gain **1.663 times the teacher**, despite high waveform correlation. The same native fitted upsampler produces gain **1.608** through the adapted residual units but **0.939** through the original teacher units.

These controlled changes show internal compatibility and downstream compensation. They do not show that allowing both sides to learn jointly is detrimental. The unchanged trained model has gain **0.969**, which is much closer to the teacher. Replacing one side of a functioning adapted interface is a different operation from continuing to optimize both sides against the final teacher waveform.

Restoring the original residual stack is also not an accepted fix: its fitted variant's waveform MAE, mel and quiet performance still trail the unchanged trained model, even though complete-group MSE improves.

## Why freezing might prevent useful recovery

The native fit from retained original teacher input coordinates reaches development MSE **0.00030431**, versus **0.00149656** from the trained student's actual inputs. The difference is not a proof of lost information: representation basis, upstream approximation and the limitations of one fixed linear fit all contribute. But it leaves an unresolved input-representation problem that freezing would prevent stages 2–3 from addressing.

Freezing would stabilize stage 4's input distribution. That can make a localization experiment easier to interpret, but stability alone is not an established improvement in the final decoder's achievable quality.

## Recommended policy and conditional diagnostic

Continue joint stages 2–4 recovery as the default, preserving the useful trained weights and the complete teacher-boundary/waveform objectives. Do not adopt the native replacement, original-stack reset or projected hints merely because a hidden-feature score improves. Keep the comparison bounded and judge actual gain, waveform/mel fidelity, quiet DC/AC residuals and near-silence behavior alongside correlation.

A prefix-freezing diagnostic would become useful if joint recovery repeatedly stalls or destabilizes, and the purpose is explicitly to test whether stage 4 can repair the residual error with the current prefix held constant. Compare a short frozen-prefix arm and a joint arm from the same snapshot, source order and recovery budget. Freezing would earn adoption only if it improves the difficult regions without sacrificing the broader output quality. The current oracle and fitting results do not meet that condition.

This recommendation does not promise that additional joint training is sufficient. If bounded recovery remains inadequate, revisit representation/compression choices rather than treating either endless training or permanent freezing as a solution by default.

Evidence: native-up4 `results/completed.json` and full development variants; `../accumulation-comparison-v1/results/completed.json`; `../projected-hints-v1/results/completed.json`. Gain is computed from the complete validation panel's `overview_window_metrics.by_source` active student/teacher energies, not only waveform snippets. No model execution or source changes were performed for this review.
