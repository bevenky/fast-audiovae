# Controlled decoder comparison

**Completed:** The waveform-only arm passed the full diagnostic at 1,800 and 1,900 additional updates. Both runs have stopped. See the [measured results and remaining gaps](results.md).

The selected reconstruction target is **0.99**, with **0.95 as an intermediate milestone**. The target is per nonquiet diagnostic crop, alongside waveform-error and amplitude checks. Quiet clips have separate noise checks. This is a waveform-reconstruction target, not a percentage of perceived quality.

The comparison has been launched on the existing H100. It uses the step-500 checkpoint from `corrected-waveform-preflight-v1`, whose SHA-256 is `df21671f774892cf5f2af0179da9fdfdc3df62ae6b195d66a11de125bb1a9fdd`.

| Setting | Waveform plus mel | Waveform only |
| --- | --- | --- |
| Parent weights, optimizer history and frozen normalization | Identical | Identical |
| Frozen encoder latents and teacher waveform targets | Exact same cached tensors | Exact same cached tensors |
| Waveform/mel gradient shares | 0.5 / 0.5 | 1.0 / 0.0 |
| Learning rate | 50-update ramp from 0.00002 to 0.0002, then constant | Same |
| Maximum additional updates | 2,000 | 2,000 |
| Evaluation / full checkpoint interval | 100 / 500 updates | Same |
| Accept diagnostic | All required checks at two consecutive evaluations | Same |

Both arms use the same ordered 32 fitted crops and 32 separate sentinel crops, derived from 16 utterances per set. The diagnostic intentionally repeats these fixed examples to establish learnability. It consumes none of the unused main-training windows. These fitted weights will not seed the representative-data pilot.

The teacher and encoder are not updated or rerun: the parent cache is opened read-only, and every cropped latent and target hash is checked against the original checkpoint's recorded inputs. The comparison code does not rescale, realign, re-encode or normalize the cached targets. Existing source, cache and checkpoint files are preserved.

The following gaps were addressed before launch:

- Parameter-gradient probes now use the full training batch instead of a two-example batch with statistics learned on 32 examples.
- Sampled diagnostics report actual optimizer weight changes, including clipping and relative update size, at the adapter, middle projection and waveform output.
- The schedule no longer decays tenfold inside the 500-update diagnostic budget.
- The 0.99 acceptance target is separate from the 0.95 milestone.
- TensorBoard shows minimum and mean nonquiet waveform cosine, passing fractions and the target, rather than only a combined loss.
- Acceptance requires two consecutive evaluations. Unseen sentinel quality is reported separately; catastrophic output instability or nonfinite training stops an arm.
- Checkpoints are saved every 500 updates to preserve a 4 GiB disk reserve. Evaluation remains every 100 updates. Interrupted comparisons are preserved and never silently restarted.

Forty-eight focused CPU checks passed locally and again in the Runpod environment. Independent code review found no remaining launch blockers. The forward model, optimizer implementation, loss definitions, discriminator implementation and gradient-balancer algorithm match the previous source snapshot; only the declared objective, schedule and measurement controls differ. New source hashes are recorded in [source-manifest.json](source-manifest.json).

The two arms run sequentially on one H100. They stop after the declared diagnostic budget and cannot automatically start adversarial training, the 20-hour pilot or a larger job.

If waveform-only succeeds where the combined objective does not, the next change is gradual mel introduction and measured balancing. If both improve, the result must be interpreted together with the additional update budget and sustained schedule. If neither fits, inspect actual updates and test normalization before redesigning the decoder. Failure within 2,000 updates alone does not establish an architectural limit.

The final quality goal still requires amplitude and quiet-noise checks, held-out multilingual and expressive evaluation, PESQ, STOI, UTMOS, DNSMOS, blind listening and streaming correctness. CPU decoder RTF will be compared only after quality qualifies.

[Live waveform progress](https://34d6pb4ub5ldrz-8888.proxy.runpod.net/#scalars&tagFilter=nonquiet_cosine)
