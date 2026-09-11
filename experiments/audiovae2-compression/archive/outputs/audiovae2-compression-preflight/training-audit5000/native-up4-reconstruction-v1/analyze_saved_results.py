"""Independent native-up4 result accounting using JSON only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
SPEC = importlib.util.spec_from_file_location("saved_json_analysis", HERE.parent / "projected-hints-v1/analyze_saved_results.py")
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)


def read(name):
    return json.loads((ROOT / name).read_text())


def endpoint(report):
    q = report["quality"]
    a = q["aggregate"]
    near = q["overview_window_metrics"]["by_source"].values()
    return {**a, "quiet_passing": a["quiet_windows"] - a["quiet_failed_windows"],
            "near_windows": sum(r["near_silence_windows"] for r in near),
            "near_passing": sum(r["near_silence_windows"] - r["near_silence_failed_windows"] for r in near)}


def compare(a, b):
    old, new = shared.source_rows(a["quality"]), shared.source_rows(b["quality"])
    out = {key: shared.paired(old, new, lambda r, k=key: r[k], higher=key == "cosine")
           for key in ("mae", "mse", "mel", "cosine", "group_mse")}
    out["quiet_rms"] = shared.paired(old, new, shared.quiet_rms)
    out["quiet_passes"] = shared.paired(old, new, lambda r: r["quiet_windows"] - r["quiet_failed"], higher=True)
    oldn, newn = (report["quality"]["overview_window_metrics"]["by_source"] for report in (a, b))
    out["near_passes"] = shared.paired(oldn, newn, lambda r: r["near_silence_windows"] - r["near_silence_failed_windows"], higher=True)
    oldg, newg = (shared.gains(report["quality"]) for report in (a, b))
    out["gain_distance"] = shared.paired(oldg, newg, lambda r: r["absolute_log_gain_error"])
    return out


def check_regions(report):
    rows = report.get("up4_local", report).get("rows", [])
    for row in rows:
        for region, phases in row["regions"].items():
            if not all(key in phases for key in ("all", "phase0", "phase1")):
                continue
            for key in ("weighted_elements", "squared_error_sum", "absolute_error_sum", "teacher_square_sum"):
                assert math.isclose(phases["all"][key], phases["phase0"][key] + phases["phase1"][key], rel_tol=1e-10, abs_tol=1e-9), (row["source_id"], region, key)


def local_comparison(before, after):
    result = {}
    for region, phases in after["aggregate"].items():
        result[region] = {}
        for phase, values in phases.items():
            old = before["aggregate"][region][phase]
            assert values["weighted_elements"] == old["weighted_elements"]
            assert values["teacher_square_sum"] == old["teacher_square_sum"]
            result[region][phase] = {"baseline_mse": old["mse"], "fitted_mse": values["mse"],
                                     "mse_change_percent": shared.relative(values["mse"], old["mse"]) if old["mse"] else None,
                                     "weighted_elements": values["weighted_elements"]}
    old_rows = {r["source_id"]: r for r in before["rows"]}
    new_rows = {r["source_id"]: r for r in after["rows"]}
    result["per_source_all_sse"] = shared.paired(old_rows, new_rows, lambda r: r["regions"]["all"]["all"]["squared_error_sum"])
    return result


def main():
    launch, completed, preservation = (read(name) for name in ("launch.json", "completed.json", "preservation.json"))
    variants = ("sliced_initial-baseline", "sliced_initial-native_fit", "sliced_initial-teacher_up4",
                "trained4625-baseline", "trained4625-native_fit", "trained4625-teacher_up4",
                "trained4625-native_fit_original_residuals", "teacher-group-end-control")
    reports = {name: read(name + ".json") for name in variants}
    endpoints = {name: endpoint(report) for name, report in reports.items()}
    initial_rows = shared.source_rows(reports["sliced_initial-baseline"]["quality"])
    invariant = ("samples", "nonquiet_samples", "quiet_samples", "quiet_windows", "teacher_rms", "mel_elements", "group_elements", "group_teacher_square_sum")
    all_checks = []
    for name, report in reports.items():
        rows = shared.source_rows(report["quality"])
        assert set(rows) == set(initial_rows)
        assert all(all(row[k] == initial_rows[sid][k] for k in invariant) for sid, row in rows.items())
        assert len(report["teacher_target_checks"]) == 96
        all_checks.extend(report["teacher_target_checks"])
        check_regions(report)
    fits, local = {}, {}
    for basis in ("sliced_initial", "trained4625"):
        fit = read(basis + "-fit.json")
        fits[basis] = fit
        assert len(fit["source_ids"]) == len(set(fit["source_ids"])) == 72
        assert not set(fit["source_ids"]) & set(initial_rows)
        assert fit["native_fold_parity"]["passed"]
        all_checks.extend(fit["teacher_target_checks"])
        train = read(basis + "-fit-local.json")
        assert len(train["rows"]) == 72
        all_checks.extend(train["teacher_target_checks"])
        check_regions(train)
        baseline = reports[basis + "-baseline"]["up4_local"]
        fitted = reports[basis + "-native_fit"]["up4_local"]
        local[basis] = {"fitted_calibration": train["aggregate"], "fitted_development": fitted["aggregate"],
                        "development_change_from_native_baseline": local_comparison(baseline, fitted)}
    selected_fit = read("teacher-retained-input-fit.json")
    selected_solver = selected_fit if "source_ids" in selected_fit else None
    recovered_solver = ROOT / "teacher-retained-input-solver-recovered.json"
    if selected_solver is None and recovered_solver.exists():
        selected_solver = read(recovered_solver.name)
        assert selected_solver["existing_fit_score_sha256"] == hashlib.sha256((ROOT / "teacher-retained-input-fit.json").read_bytes()).hexdigest()
        assert selected_solver["original_launch_sha256"] == hashlib.sha256((ROOT / "launch.json").read_bytes()).hexdigest()
        assert selected_solver["source_ids"] == fits["sliced_initial"]["source_ids"]
    if selected_solver is not None:
        all_checks.extend(selected_solver["teacher_target_checks"])
    selected = {}
    for name, count in (("fit", 72), ("development", 96)):
        report = read("teacher-retained-input-" + name + ".json")
        assert len(report["rows"]) == count
        selected[name] = report["aggregate"]
        all_checks.extend(report["teacher_target_checks"])
        check_regions(report)
    pairs = {}
    for basis in ("sliced_initial", "trained4625"):
        for variant in ("native_fit", "teacher_up4"):
            pairs[basis + "-" + variant + "-versus-baseline"] = compare(reports[basis + "-baseline"], reports[basis + "-" + variant])
    pairs["trained-original-residuals-versus-adapted-residuals"] = compare(reports["trained4625-native_fit"], reports["trained4625-native_fit_original_residuals"])
    pairs["trained-native-fit-original-residuals-versus-baseline"] = compare(reports["trained4625-baseline"], reports["trained4625-native_fit_original_residuals"])
    amplitude = {name: shared.gain_summary(shared.gains(report["quality"])) for name, report in reports.items()}
    integrity = {"completed": completed["completed"], "preservation": preservation,
                 "teacher_checks": len(all_checks), "all_teacher_checks_bitwise": all(r["bitwise_equal"] for r in all_checks),
                 "original_saved_teacher_checks": len(all_checks) - (len(selected_solver["teacher_target_checks"]) if recovered_solver.exists() else 0),
                 "receipt_recovery_teacher_checks": len(selected_solver["teacher_target_checks"]) if recovered_solver.exists() else 0,
                 "all_teacher_checks_pass": all(r["passed"] for r in all_checks),
                 "max_teacher_cache_error": max(r["max_abs"] for r in all_checks),
                 "all96_source_and_sample_support_identical": True, "phase_decomposition_counts_and_sums_match": True,
                 "no_neural_training": completed["no_neural_training"], "automatic_promotion": completed["automatic_promotion"],
                 "fit_source_ids_equal": fits["sliced_initial"]["source_ids"] == fits["trained4625"]["source_ids"] == [r["source_id"] for r in read("teacher-retained-input-fit.json")["rows"]],
                 "teacher_retained_solver_receipt_available": selected_solver is not None,
                 "teacher_retained_solver_recovered_separately": recovered_solver.exists(),
                 "fit_development_disjoint": True,
                 "native_fold_parity": {k: r["native_fold_parity"] for k, r in fits.items()}}
    result = {"integrity": integrity, "endpoints": endpoints, "local_reconstruction": local,
              "teacher_retained_input_local_control": selected, "paired_quality": pairs, "amplitude": amplitude,
              "waveform_regions": {name: r["waveform_regions"] for name, r in reports.items()},
              "solver_reports": fits, "teacher_retained_solver": selected_solver,
              "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(ROOT.rglob("*.json")) if not p.name.startswith("independent")}}
    (ROOT / "independent-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Native upsampler reconstruction: independent audit", "",
             "**Retain the existing trained checkpoint. None of the fitted replacements improves its audio quality.** This experiment does establish a concrete mechanism: the adapted stage4 residual stack expects its jointly learned upstream representation. Substituting an original-teacher representation into it can produce a large gain error even when local features or waveform correlation look better.", "",
             "All calculations below use saved JSON reports only. Native fitting uses72 calibration sources and a fixed shared-bias causal stride2/kernel4 operator;96 development sources are evaluated without fitting on them. Original initialization and the retained trained4625 checkpoint are separate bases. Teacher injections are diagnostic controls, not deployable replacements.", "",
             f"Integrity: {integrity['original_saved_teacher_checks']} original saved teacher/cache checks plus{integrity['receipt_recovery_teacher_checks']} receipt-recovery checks; all bitwise={integrity['all_teacher_checks_bitwise']}; all source/window support invariant; both native-fold checks pass. No neural training, original checkpoint mutation or automatic promotion occurred. The teacher-retained-input solver receipt was originally overwritten by its calibration score, then recovered in a separate identical72-source fit without new development scoring; both original score and launch hashes remain verified.", "",
             "## Final waveform on the common96 sources", "", "| Variant | MAE | MSE | Active cosine | Mel | Group MSE | Quiet passes | Near passes | Peak |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, a in endpoints.items():
        lines.append(f"| {name} | {a['mae']:.9g} | {a['mse']:.9g} | {a['nonquiet_cosine_mean']:.9g} | {a['mel']:.9g} | {a['group_mse']:.9g} | {a['quiet_passing']}/{a['quiet_windows']} | {a['near_passing']}/{a['near_windows']} | {a['peak_abs_max']:.7g} |")
    lines += ["", "Peaks are descriptive, not directional quality scores. Quiet, near-silence and active labels use the unchanged teacher masks.", "",
              "## Local native-up4 output reconstruction", "", "| Basis | Region | Calibration MSE | Development MSE | Development change vs native baseline |", "|---|---|---:|---:|---:|"]
    for basis, r in local.items():
        for region in ("all", "active", "quiet", "near_silence", "startup_40ms"):
            a, b = r["fitted_calibration"][region]["all"], r["fitted_development"][region]["all"]
            change = r["development_change_from_native_baseline"][region]["all"]["mse_change_percent"]
            lines.append(f"| {basis} | {region} | {a['mse']:.9g} | {b['mse']:.9g} | {change:.4f}% |")
    lines += ["", "The JSON separately reports phase0 and phase1, source-level SSE changes, weighted support, and teacher energy. A reduced local feature error is not itself an audio-quality gain.", "",
              "## Teacher retained-input control", "", "| Region | Fit MSE | Held-out MSE | Held-out SSE / teacher energy |", "|---|---:|---:|---:|"]
    for region in ("all", "active", "quiet", "near_silence", "startup_40ms"):
        a, b = selected["fit"][region]["all"], selected["development"][region]["all"]
        lines.append(f"| {region} | {a['mse']:.9g} | {b['mse']:.9g} | {b['relative_squared_error']:.9g} |")
    lines += ["", "This last control uses original teacher retained coordinates. Its fit is never installed into the adapted model; it measures local predictability without accumulated student drift.", "",
              "## Paired source directions", "", "| Comparison | Metric | Better | Worse | Equal |", "|---|---|---:|---:|---:|"]
    for comparison, metrics in pairs.items():
        for metric in ("mae", "mel", "cosine", "group_mse", "quiet_rms", "quiet_passes", "near_passes", "gain_distance"):
            counts = metrics[metric]["counts"]
            lines.append(f"| {comparison} | {metric} | {counts['better']} | {counts['worse']} | {counts['equal']} |")
    lines += ["", "Directions retain every measured difference. Existing numerical reproduction tolerances are recorded separately in the JSON and are not perceptual acceptance thresholds.", "",
              "## Mechanism and decision", "",
              "1. **The original downstream decoder is sufficient when given the right internal signal.** Injecting the exact teacher up4 output into the sliced initialization recovers waveform MAE4.80e-9, all2,544 quiet passes and all184 near-silence passes. Injecting the teacher group-end into the unchanged suffix gives exactly zero waveform and mel error on all96 sources. These are controlled teacher-feature interventions, not deployable paths.",
              "2. **The trained internal coordinates are no longer interchangeable with the original teacher's.** Injecting exact teacher up4 output into the adapted residual stack gives active cosine0.993828 but waveform MAE0.019783 and pooled active RMS1.663 times the teacher. All94 active recordings have excess RMS. The signed teacher-axis gain, sum(prediction×teacher)/sum(teacher²), is0.937 in the baseline,1.521 with native fitting,1.641 with exact up4 injection, and0.902 with native fitting plus original residuals. Exact local feature matching therefore does not imply correct final amplitude, and0.99 correlation alone would misleadingly approve this variant.",
              "3. **The changed residual stack causally mediates much of this splice failure.** Keep the fitted native operator and all its upstream inputs fixed; restore only the three original stage4 residual units. MAE falls75.91%, from0.020350 to0.004903, with improvement on all96 sources. The pooled active RMS ratio falls from1.608 to0.939. Because this intervention changes only that residual stack, this is direct evidence of downstream coadaptation, not just an association between two loss curves.",
              "4. **That control is not a quality fix.** Relative to the retained trained baseline, fit plus original residuals still has15.79% higher MAE,37.09% higher mel error and9.35% higher quiet RMS. MAE worsens95/96 sources, mel95/96 and active cosine94/94. Quiet passes fall138→89; near-silence passes18→5. Group-boundary MSE improves39.13% and every source, again demonstrating why that feature statistic cannot replace waveform quality.",
              "5. **The native linear fit works mathematically, but its available student features remain an imperfect predictor.** Local up4 SSE improves86.93% at initialization and81.91% at the trained checkpoint, on all96 sources in each case. Both phases improve, including quiet and near-silence regions. At the trained checkpoint, held-out fitted MSE0.00149656 remains4.92 times the teacher-retained-input control0.000304315; the quiet gap is4.21 times and the near-silence gap9.93 times. Fit and development errors are comparable. Both actual-student solvers have full rank512, tiny normal-equation residuals, and passing folded-native parity. This is evidence of limited predictability within this fixed native linear operator, not proof that the upstream student has irreversibly lost the information or cannot learn a nonlinear alternative.",
              "6. **This does not uniquely explain the preceding projected-hint A/B.** Those hints compared trainable readouts of student features with teacher features; they did not inject raw teacher features into the decoder. The present experiment establishes raw-splice risk and internal coadaptation. It does not identify the unique cause of the projected-hint regression or the earlier training trajectory's gain drift.", "",
              "No runtime layer was added, no optimizer update was made, and no CPU timing was measured. A future recipe that preserves original downstream coordinates from its own starting point would require a separately approved training comparison. These results do not justify patching or restarting the accepted model automatically."]
    (HERE / "independent-analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"integrity": integrity, "endpoints": endpoints,
                      "paired_counts": {name: {key: row['counts'] for key, row in values.items()} for name, values in pairs.items()}}, indent=2))


if __name__ == "__main__":
    main()
