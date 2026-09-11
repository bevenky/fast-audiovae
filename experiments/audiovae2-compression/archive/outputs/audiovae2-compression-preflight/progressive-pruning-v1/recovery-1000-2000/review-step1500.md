# Same-width recovery at update 1,500

This is the scheduled review of joint stage 2 through 4 recovery at widths 384/256 against the unchanged original AudioVAE2 teacher. It uses the same 96 held-out recordings and source ledger. No changes were made in response to this interim review.

Compared with update 1,000, active waveform cosine improved from 0.988697 to 0.991630, waveform MAE from 0.003286 to 0.002671, mel error from 0.201765 to 0.173676, and group MSE from 0.003539 to 0.002741. Active pooled RMS recovered from 93.65% to 97.90% of the teacher. Waveform MAE improved on 95 of 96 recordings.

Quiet windows passing increased from 358 to 1,000 out of 2,544. Near-silence remained 167/184. Startup first-20-ms windows remained 0/13. Sustained source-silence passes fell from 164 to 161/164, and interior near-silence from 49 to 46/50, despite lower residual RMS. These were strict output-amplitude threshold crossings, so both continuous errors and pass counts must remain visible.

Whistle reconstruction was less consistent than the pooled metrics. Retain its individual waveform and RMS measurements for the final review. Continue unchanged only to the already approved update 2,000, then stop and assess.
