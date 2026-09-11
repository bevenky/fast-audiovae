"""Read only completed JSON reports; never import a model or training runtime."""
from pathlib import Path
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"


def read(path):
    return json.loads(path.read_text())


def relative(new, old):
    return 100 * (new / old - 1) if old else None


def paired(before, after, metric, higher=False):
    rows = []
    for sid, a in before.items():
        old, new = metric(a), metric(after[sid])
        if old is None or new is None:
            continue
        direction = (new - old) * (-1 if higher else 1)
        rows.append({"source_id": sid, "before": old, "after": new,
                     "change": new - old, "change_percent": relative(new, old),
                     "outcome": "worse" if direction > 0 else "better" if direction < 0 else "equal",
                     "within_existing_numerical_reproduction_tolerance": math.isclose(old, new, rel_tol=1e-5, abs_tol=1e-6)})
    return {"eligible_sources": len(rows),
            "counts": {key: sum(r["outcome"] == key for r in rows) for key in ("better", "worse", "equal")},
            "beyond_numerical_tolerance": {key: sum(r["outcome"] == key and not r["within_existing_numerical_reproduction_tolerance"] for r in rows)
                                          for key in ("better", "worse", "equal")},
            "rows": rows}


def source_rows(report):
    rows = {r["source_id"]: r for r in report["rows"]}
    assert len(rows) == len(report["rows"]) == 96
    return rows


def quiet_rms(row):
    return math.sqrt(row["quiet_error_sum"] / row["quiet_samples"]) if row["quiet_samples"] else None


def gains(report):
    result = {}
    for sid, r in report["overview_window_metrics"]["by_source"].items():
        te, pe = r["active_teacher_energy"], r["active_student_energy"]
        if te > 0:
            result[sid] = {"gain": math.sqrt(pe / te), "teacher_energy": te, "student_energy": pe}
            result[sid]["absolute_log_gain_error"] = abs(math.log(result[sid]["gain"]))
    return result


def gain_summary(rows):
    values = [r["gain"] for r in rows.values()]
    return {"eligible_sources": len(rows), "pooled_rms_gain": math.sqrt(sum(r["student_energy"] for r in rows.values()) / sum(r["teacher_energy"] for r in rows.values())),
            "median_gain": statistics.median(values), "mean_gain": statistics.mean(values),
            "below_teacher": sum(g < 1 for g in values), "above_teacher": sum(g > 1 for g in values),
            "mean_absolute_log_gain_error": statistics.mean(r["absolute_log_gain_error"] for r in rows.values()),
            "rows": rows}


def trajectories(snapshots):
    assert [r["sources_consumed"] for r in snapshots] == list(range(0, 1501, 60))
    assert [r["optimizer_step"] for r in snapshots] == list(range(4625, 4751, 5))
    ids = [r["source_id"] for r in snapshots[0]["cases"]]
    assert len(ids) == len(set(ids)) == 4
    out = {}
    for sid in ids:
        rows = [next(r for r in s["cases"] if r["source_id"] == sid) for s in snapshots]
        g = [r["active_rms_gain"] for r in rows]
        signs = [1 if b > a else -1 for a, b in zip(g, g[1:]) if a != b]
        out[sid] = {"observations": len(g), "mean_absolute_log_gain_error": statistics.mean(abs(math.log(v)) for v in g),
                    "gain_population_std": statistics.pstdev(g), "gain_min": min(g), "gain_max": max(g),
                    "gain_mean": statistics.mean(g), "gain_total_variation": sum(abs(b - a) for a, b in zip(g, g[1:])),
                    "gain_reversals": sum(a != b for a, b in zip(signs, signs[1:])),
                    "mean_mae": statistics.mean(r["mae"] for r in rows), "initial": rows[0], "final": rows[-1]}
    for name, selected in (("three_speech_recordings", [sid for sid in ids if not sid.startswith("freesound:")]),
                           ("one_whistle_recording", [sid for sid in ids if sid.startswith("freesound:")])):
        out[name] = {"sources": selected, "observations_per_source": 26,
                     **{key: statistics.mean(out[sid][key] for sid in selected)
                        for key in ("mean_absolute_log_gain_error", "gain_population_std", "mean_mae")}}
    return out


def endpoint(report, coefficients):
    a = report["aggregate"]
    o = report["overview_window_metrics"]
    near = o["by_source"]
    return {**a, "quiet_passing_windows": a["quiet_windows"] - a["quiet_failed_windows"],
            "near_silence_windows": sum(r["near_silence_windows"] for r in near.values()),
            "near_silence_passing_windows": sum(r["near_silence_windows"] - r["near_silence_failed_windows"] for r in near.values()),
            "original_weighted_objective": coefficients["waveform"] * a["mae"] + coefficients["mel"] * a["mel"] + coefficients["feature"] * a["group_mse"]}


def main():
    completed, launch, preservation = (read(ROOT / name) for name in ("completed.json", "launch.json", "preservation.json"))
    arms = ("baseline", "two_hints")
    reports = {arm: {tag: read(ROOT / arm / f"development-source{n}.json") for tag, n in (("initial", 0), ("final", 1500))} for arm in arms}
    endpoints = {"initial": endpoint(reports["baseline"]["initial"], launch["coefficients"]),
                 **{arm: endpoint(reports[arm]["final"], launch["coefficients"]) for arm in arms}}
    all_reports = [reports[arm][tag] for arm in arms for tag in ("initial", "final")]
    rows = {arm: source_rows(reports[arm]["final"]) for arm in arms}
    initial_rows = source_rows(reports["baseline"]["initial"])
    invariant_keys = ("samples", "nonquiet_samples", "quiet_samples", "quiet_windows", "teacher_rms",
                      "mel_elements", "group_elements", "group_teacher_square_sum")
    support_same = all(all(r[k] == initial_rows[r["source_id"]][k] for k in invariant_keys)
                       for report in all_reports for r in report["rows"])
    overview_invariants = ("active_teacher_energy", "near_silence_windows", "quiet_windows", "valid_samples")
    first_overview = all_reports[0]["overview_window_metrics"]["by_source"]
    support_same &= all(all(row[k] == first_overview[sid][k] for k in overview_invariants)
                        for report in all_reports for sid, row in report["overview_window_metrics"]["by_source"].items())
    ledger = {}
    for arm in arms:
        logs = [json.loads(line) for line in (ROOT / arm / "train.jsonl").read_text().splitlines()]
        ids = [sid for record in logs for sid in record["source_ids"]]
        assert ids == launch["source_ids"] and len(ids) == len(set(ids)) == 1500
        assert len(logs) == 125 and all(len(r["source_ids"]) == 12 for r in logs)
        assert [r["optimizer_step"] for r in logs] == list(range(4626, 4751))
        assert [r["sources_consumed"] for r in logs] == list(range(12, 1501, 12))
        cache = [c for record in logs for c in record["teacher_cache_checks"]]
        assert len(cache) == 1500
        restored = read(ROOT / arm / "restoration.json")
        ledger[arm] = {"updates": len(logs), "sources": len(ids), "final_optimizer_step": logs[-1]["optimizer_step"],
                       "scored_samples": logs[-1]["scored_samples"], "source_order_matches_launch": True,
                       "all_targets_bitwise": all(c["bitwise_equal"] for c in cache),
                       "all_targets_original_tolerance": all(c["allclose_original_tolerance"] for c in cache),
                       "exact_restoration": all(c["equal"] for c in restored["state"].values()),
                       "saved_initial_quality": restored["saved_quality"], "paired_initial_quality": restored["paired_quality"]}
    assert support_same
    assert ledger["baseline"]["scored_samples"] == ledger["two_hints"]["scored_samples"]
    pair_metrics = {key: paired(rows["baseline"], rows["two_hints"], lambda r, k=key: r[k], higher=key == "cosine")
                    for key in ("mae", "mse", "cosine", "mel", "mel_linear", "mel_log", "group_mse")}
    pair_metrics["quiet_rms"] = paired(rows["baseline"], rows["two_hints"], quiet_rms)
    pair_metrics["quiet_passes"] = paired(rows["baseline"], rows["two_hints"], lambda r: r["quiet_windows"] - r["quiet_failed"], higher=True)
    near_rows = {arm: reports[arm]["final"]["overview_window_metrics"]["by_source"] for arm in arms}
    pair_metrics["near_silence_passes"] = paired(near_rows["baseline"], near_rows["two_hints"],
                           lambda r: r["near_silence_windows"] - r["near_silence_failed_windows"], higher=True)
    from_start = {arm: {key: paired(initial_rows, rows[arm], lambda r, k=key: r[k], higher=key == "cosine")
                       for key in ("mae", "cosine", "mel", "group_mse")} for arm in arms}
    amplitude = {"initial": gain_summary(gains(reports["baseline"]["initial"])),
                 **{arm: gain_summary(gains(reports[arm]["final"])) for arm in arms}}
    amplitude["paired_gain_distance"] = paired(amplitude["baseline"]["rows"], amplitude["two_hints"]["rows"], lambda r: r["absolute_log_gain_error"])
    trajectory = {arm: trajectories(read(ROOT / arm / "case-trajectories.json")) for arm in arms}
    h = {arm: {tag: read(ROOT / arm / f"hints-source{n}.json") for tag, n in (("initial", 0), ("final", 1500))} for arm in arms}
    hint_changes = {}
    for arm in arms:
        hint_changes[arm] = {}
        for name in ("stage3_up", "stage4_up"):
            initial = h[arm]["initial"]["aggregate"]["initial_frozen"][name]
            current = h[arm]["final"]["aggregate"]["current"][name]
            frozen = h[arm]["final"]["aggregate"]["initial_frozen"][name]
            old = {r["source_id"]: r for r in h[arm]["initial"]["rows"]}
            new = {r["source_id"]: r for r in h[arm]["final"]["rows"]}
            assert len(old) == len(new) == 96 and set(old) == set(initial_rows)
            hint_changes[arm][name] = {"initial": initial, "final_current_projector": current, "final_initial_frozen_projector": frozen,
                                     "current_change_percent": relative(current, initial), "fixed_change_percent": relative(frozen, initial),
                                     "current_vs_fixed_change_percent": relative(current, frozen),
                                     "fixed_readout_per_source": paired(old, new, lambda r, k=name: r["initial_frozen"][k]),
                                     "current_readout_per_source": paired(old, new, lambda r, k=name: r["current"][k])}
    expressive = {}
    for arm in arms:
        selected = [r for sid, r in rows[arm].items() if sid.startswith(("freesound:", "thorsten_emotional:whisper/"))]
        n = sum(r["samples"] for r in selected)
        expressive[arm] = {"recording_level_expressive_sources": len(selected),
                           "selection": "Ten Freesound recordings and one named Thorsten whisper recording; recording labels, not measured event-time occupancy",
                           "samples": n,
                           "sample_pooled_mae": sum(r["mae"] * r["samples"] for r in selected) / n,
                           "mean_active_cosine": statistics.mean(r["cosine"] for r in selected if r["cosine"] is not None),
                           "source_ids": [r["source_id"] for r in selected]}
    integrity = {"completed_comparison_valid": completed["comparison_valid"], "preservation": preservation,
                 "initial_reports_exact_equal": reports["baseline"]["initial"] == reports["two_hints"]["initial"],
                 "fixed_teacher_and_sample_window_coverage": support_same,
                 "calibration_sources": len(launch["calibration_ids"]), "development_sources": len(launch["development_ids"]),
                 "fit_development_disjoint": not set(launch["calibration_ids"]) & set(launch["development_ids"]),
                 "arms": ledger, "automatic_promotion": completed["automatic_promotion"]}
    result = {"integrity": integrity, "endpoints": endpoints, "paired_sources": pair_metrics,
              "from_start": from_start, "amplitude": amplitude, "trajectories": trajectory,
              "hint_readouts": hint_changes, "expressive_recordings": expressive,
              "original_objective_coefficients": launch["coefficients"], "hint_coefficients": launch["hint_coefficients"],
              "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(ROOT.rglob("*.json")) if not p.name.startswith("independent")}}
    (ROOT / "independent-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Projected hints: independent saved-result audit", "",
             "**Do not adopt these two hint losses as a quality improvement from this experiment.** They make intermediate teacher matching better, including under the original frozen readouts, but the common waveform and spectral endpoint does not improve. Quiet pass counts and the prespecified whistle trajectory worsen. Both experimental checkpoints remain useful evidence; neither was automatically promoted.", "",
             "The comparison uses the retained accumulation12 checkpoint at optimizer step4625. Both arms consume the same1,500 ordered diagnostic sources with125 updates, ending at4750. Source reuse across diagnostic arms is explicit. These are development-panel results, not an untouched final test.", "",
             f"Integrity: comparison_valid={completed['comparison_valid']}; initial reports exactly equal={integrity['initial_reports_exact_equal']}; teacher/window coverage invariant={support_same}. Each arm has {ledger['baseline']['scored_samples']:,} scored training samples. All teacher/cache targets bitwise equal: baseline={ledger['baseline']['all_targets_bitwise']}, hints={ledger['two_hints']['all_targets_bitwise']}.", "",
             "## Common96-source endpoint", "", "| Metric | Initial | Baseline | Two hints |", "|---|---:|---:|---:|"]
    for key in ("mae", "mse", "nonquiet_cosine_mean", "mel", "mel_linear", "mel_log", "group_mse", "original_weighted_objective", "quiet_residual_rms_mean", "quiet_passing_windows", "near_silence_passing_windows", "peak_abs_max", "overshoot_samples"):
        lines.append("| " + key + " | " + " | ".join(f"{endpoints[arm][key]:.9g}" for arm in ("initial", "baseline", "two_hints")) + " |")
    lines += ["", f"MAE change versus baseline: {relative(endpoints['two_hints']['mae'], endpoints['baseline']['mae']):.4f}%. Original weighted-objective change: {relative(endpoints['two_hints']['original_weighted_objective'], endpoints['baseline']['original_weighted_objective']):.4f}%. Quiet support={endpoints['baseline']['quiet_windows']}; near-silence support={endpoints['baseline']['near_silence_windows']}.", "",
              "## Paired sources", "", "Counts retain every measured direction; the JSON separately marks differences within existing numerical reproduction tolerance. That tolerance is not a perceptual acceptance criterion.", "",
              "| Metric | Better | Worse | Equal |", "|---|---:|---:|---:|"]
    for key, row in pair_metrics.items():
        lines.append(f"| {key} | {row['counts']['better']} | {row['counts']['worse']} | {row['counts']['equal']} |")
    lines += ["", "## Full-panel active amplitude", "", "| Metric | Initial | Baseline | Two hints |", "|---|---:|---:|---:|"]
    for key in ("eligible_sources", "pooled_rms_gain", "median_gain", "below_teacher", "above_teacher", "mean_absolute_log_gain_error"):
        lines.append("| " + key + " | " + " | ".join(f"{amplitude[arm][key]:.9g}" for arm in ("initial", "baseline", "two_hints")) + " |")
    lines += ["", "## All26 matched trajectory points", "", "The panel contains three speech recordings and one whistling recording. It does not represent four expressive categories. Every predefined snapshot is included, including the common starting point.", "",
              "| Cohort | Measure | Baseline | Two hints | Change |", "|---|---|---:|---:|---:|"]
    for cohort in ("three_speech_recordings", "one_whistle_recording"):
        for key in ("mean_absolute_log_gain_error", "gain_population_std", "mean_mae"):
            a, b = (trajectory[arm][cohort][key] for arm in arms)
            lines.append(f"| {cohort} | {key} | {a:.9g} | {b:.9g} | {relative(b,a):.4f}% |")
    lines += ["", "## Readout interpretation", "", "| Arm | Hint | Initial | Final learned readout | Final original frozen readout |", "|---|---|---:|---:|---:|"]
    for arm in arms:
        for name, row in hint_changes[arm].items():
            lines.append(f"| {arm} | {name} | {row['initial']:.9g} | {row['final_current_projector']:.9g} | {row['final_initial_frozen_projector']:.9g} |")
    lines += ["", "A learned-readout decrease can include changes in both the student representation and the auxiliary map. The frozen-initial-readout score holds the map fixed, but still does not establish downstream waveform quality. Common waveform, spectral, amplitude and quiet metrics remain the decision criteria. The readouts are external to the decoder and add no deployed inference operations.", "",
              "## What the paired result establishes", "",
              "- Internal progress is not merely the trainable projectors absorbing error. With both original readouts frozen for scoring, stage3 hint MSE improves9.58% and stage4 improves13.86% versus the initial checkpoint. Fixed-readout loss improves on96/96 and94/96 sources respectively. The directly measured full group-boundary MSE also improves4.04% versus baseline, on85/96 sources.",
              "- That improvement does not carry through to final reconstruction: waveform MAE is worse on55/96 sources and common mel error on74/96. The mean active correlation is effectively unchanged, with48 sources improving and46 worsening. The original weighted objective also worsens slightly. The run therefore supplies evidence against using these particular hidden-state MSE hints as a proxy for waveform quality.",
              "- Quiet residual RMS improves only0.292% when pooled; it worsens on39/68 sources with quiet support. Spanish quiet RMS rises12.90%, Freesound236508 rises10.74%, and Welsh rises8.11%. Twelve recordings lose quiet passing windows while three gain some. The two lost near-silence passes are both in Freesound277554. Both arms lose near-silence passes compared with the starting checkpoint:18 initially,3 baseline,1 hints. Thresholds and window coverage are unchanged.",
              "- Full-panel amplitude distance improves on61/94 active sources and worsens on33. This is a limited favorable result. It does not contradict the worse three-speech-recording temporal trajectory, which measures a different subset over all26 points. The whistle finishes at86.43% of teacher RMS versus86.94% for baseline, with1.58% worse endpoint MAE; its trajectory mean MAE worsens3.01%. All61 of its quiet windows still fail in both arms.",
              f"- Across ten Freesound recordings plus the named Thorsten whisper recording, sample-pooled MAE changes from{expressive['baseline']['sample_pooled_mae']:.9g} to{expressive['two_hints']['sample_pooled_mae']:.9g}; mean active correlation changes from{expressive['baseline']['mean_active_cosine']:.9g} to{expressive['two_hints']['mean_active_cosine']:.9g}. These are recording-level labels, not measurements of event-time occupancy.",
              "- No full-scale overshoots occur in either arm. Maximum sample level is reported descriptively; a smaller peak is not automatically better fidelity.", "",
              "This is one125-update, paired-source diagnostic, not a proof that all feature distillation fails or that the narrowed decoder has reached a capacity floor. It supports keeping the present hint recipe out of the accepted training path. It does not justify a speculative coefficient change or another architecture change without a separate proposal.", "",
              "The machine-readable audit includes every paired source, cohort, preserved input hash and integrity check. No model inference or training was executed by this analysis."]
    (HERE / "independent-analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"integrity": integrity, "endpoints": endpoints,
                      "paired_counts": {k: v["counts"] for k, v in pair_metrics.items()},
                      "amplitude_counts": amplitude["paired_gain_distance"]["counts"],
                      "hint_readouts": {arm: {name: {k:v for k,v in row.items() if not k.endswith("per_source")} for name,row in values.items()} for arm,values in hint_changes.items()}}, indent=2))


if __name__ == "__main__":
    main()
