# Full development loss calibration

The exact original teacher scores **21.6040** on the same 2,839 development clips and leading crops used by the final evaluation. The final student scores **25.1725**. This explains why a total near 22 is not equivalent to a near-zero reconstruction error, but it does not establish that the student's waveform is close to the teacher.

| Component | Weight | Exact teacher prediction | Final student |
| --- | ---: | ---: | ---: |
| Teacher spectral error | 15 | 0 | 0.2487891 |
| Teacher waveform L1 | 1 | 0 | 0.0299166 |
| Original-reference spectral error | 45 | 0.4800891 | 0.4757936 |
| Weighted total | | **21.6040085** | **25.1724646** |

The exact teacher's total comes entirely from the original-reference term. Its two teacher-matching terms are zero by construction. The student's slightly lower original-reference term saves about 0.1933 weighted loss, while its teacher spectral and waveform errors add about 3.7618. The final total is therefore about 3.5685 above the teacher baseline.

**21.604 is a measured teacher baseline, not a mathematical global minimum of this objective.** A different waveform can trade teacher fidelity against the more heavily weighted original reference, potentially scoring below the teacher. Consequently, neither reaching nor beating this number would prove teacher-quality audio.

The full-dev baseline used one CPU thread, already existing FP32 caches and no model or teacher forward calls. It verified every cache's indexed file hash, tensor hashes, source identity and teacher identity. All 2,839 clips were present. It used the same first 64 latent frames, valid-tail trimming and arithmetic mean over utterances as the final evaluator: 293,382,780 scored 48 kHz samples in total. Teacher terms were set to their analytical zero; the remaining reference branch uses the actual loss implementation. On the first clip its result matched the complete criterion exactly. No training, cache changes, or GPU operations were performed.

The final student's numbers above were supplied from the separately inspected completed-run evaluation. They were not recomputed by this teacher-only diagnostic.

## Development-set coverage

| Dataset | Clips | Exact-teacher total |
| --- | ---: | ---: |
| EmoGator | 1,530 | 19.7535 |
| CREMA-D | 738 | 28.4691 |
| JVNV | 426 | 17.3738 |
| JNV | 105 | 15.8932 |
| LibriSpeech | 25 | 26.4867 |
| FSD50K vocal, CC-BY-3.0 | 10 | 29.0938 |
| FSD50K vocal, CC0 | 3 | 18.4160 |
| Human whistle, CC0 | 1 | 12.2280 |
| Human whistle, CC-BY-4.0 | 1 | 10.1139 |

The language labels are English 763, Japanese 531 and unspecified 1,545. There are no FLEURS or IndicVoices development rows in this evaluation. This is predominantly an expressive-vocalization development set; its aggregate cannot establish multilingual speech reconstruction quality. The group table describes its composition and uses each group's own mean, while the overall 21.6040 remains the per-utterance mean across all clips.

Machine-readable evidence: [full-dev-teacher-baseline.json](full-dev-teacher-baseline.json). It includes every per-clip value, group means, manifest hash, original-teacher provenance, implementation hashes and zero-missing accounting.
