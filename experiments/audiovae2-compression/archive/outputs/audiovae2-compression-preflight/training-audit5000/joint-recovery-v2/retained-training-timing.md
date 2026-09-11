# Retained training time through step 6,000

The retained lineage has **65.635 minutes of recorded segment elapsed time** through optimizer step 6,000, including **54.342 minutes through step 5,625**. These are run-clock totals, not isolated GPU compute. They exclude discarded branches and preparation/gaps outside the selected run timers.

| Retained optimizer steps | Updates | Sources per update | Observed segment elapsed |
|---|---:|---:|---:|
| 0–256 | 256 | 3 | 0.609 min |
| 256–1,000 | 744 | 3 | 1.864 min |
| 1,000–4,500 | 3,500 | 3 | 25.342 min |
| 4,500–4,625 | 125 | 12 | 1.933 min |
| 4,625–5,625 | 1,000 | 12 | 24.594 min |
| 5,625–6,000 | 375 | 12 | 11.293 min |

The first selected 1,000 updates took 148.404 seconds on the recorded run clocks. The original continuation through step 4,500 ran from its first recorded update at 09:23:28 UTC to step 4,500 at 09:48:43 UTC on September 10. The branch beyond step 4,500 was discarded and is not counted here.

Exact active-compute time cannot be reconstructed for the full lineage because early `step_seconds` fields were sampled. The later runs record all update sections: 1,160.862 seconds for 4,625–5,625 and 445.169 seconds for 5,625–6,000. These sections include teacher inference, student forward/backward, optimizer and target checks; they are not GPU-kernel-only measurements. No estimated early active total is included.

Timing boundaries are not identical: early journal endpoints precede final scoring, while completed arm receipts include final checks. The table is therefore an explicit sum of observed intervals, not a precision claim about one continuous training clock.
