"""Read-only seven-scale development evaluation of the four completed arms.

No optimizer is created. The old common spectral metric must reproduce both
its saved aggregate and every source before the addendum is marked complete.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

import settings_screen as screen
from author_mel import AuthorMelConfig, AuthorMelLoss

base = screen.base
VERSION = "audiovae2_reference_mel_addendum_v1"
COMMON_ATOL = 1e-6
COMMON_RTOL = 1e-5


def read_json(path):
    return json.loads(Path(path).read_text())


def load_authenticated_arm(directory, arm, identity):
    """Authenticate bytes and training identity before returning any weights."""
    if arm not in {row[0] for row in screen.ARMS}:
        raise ValueError("Unknown completed screen arm")
    directory = Path(directory)
    completion = read_json(directory/"completed.json")
    launch = read_json(directory/"launch.json")
    _, definition, rate = next(row for row in screen.ARMS if row[0] == arm)
    if (completion.get("arm") != arm or completion.get("step") != screen.UPDATES
            or completion.get("unique_sources") != screen.UPDATES*screen.SOURCES_PER_UPDATE
            or completion.get("frozen_decoder_state_preserved") is not True
            or completion.get("automatic_promotion") is not False):
        raise ValueError("Expected the completed, unpromoted 256-update arm")
    if (launch.get("arm") != arm or launch.get("definition") != definition
            or launch.get("learning_rate") != rate
            or launch.get("screen_identity_sha256") != screen.digest(identity)
            or launch.get("original_identity") != identity["original_identity"]
            or launch.get("initial_group_sha256") != identity["initial_group_sha256"]
            or launch.get("channel_selection") != identity["channel_selection"]
            or launch.get("fit_source_ids_sha256") != identity["fit_source_ids_sha256"]
            or launch.get("optimizer_initial_state_entries") != 0):
        raise ValueError("Arm launch identity does not match the paired screen")
    path = directory/"final.pt"
    if base.sha(path) != completion.get("checkpoint_sha256"):
        raise ValueError("Completed checkpoint bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (payload.get("format") != screen.VERSION or payload.get("step") != screen.UPDATES
            or payload.get("fit_cursor") != len(identity["fit_source_ids"])
            or payload.get("sources_seen") != identity["fit_source_ids"]
            or payload.get("identity") != launch):
        raise ValueError("Checkpoint state does not match its authenticated launch")
    if not isinstance(payload.get("group"),dict) or not payload["group"]:
        raise ValueError("Checkpoint group weights are missing")
    return payload


def authenticate_screen(args):
    metadata, selection, initial, preflight, manifest, pools = screen.authenticate_inputs(args)
    directory = args.screen_out
    identity = read_json(directory/"screen-identity.json")
    completion = read_json(directory/"completed.json")
    expected_arms = [row[0] for row in screen.ARMS]
    if (identity.get("version") != screen.VERSION
            or identity.get("original_preflight") != metadata
            or identity.get("original_identity") != preflight["identity"]
            or identity.get("channel_selection") != selection
            or identity.get("screen_sha256") != base.sha(screen.__file__)
            or identity.get("author_mel_sha256") != base.sha(Path(__file__).with_name("author_mel.py"))
            or identity.get("development_source_ids") != [r["source_id"] for r in pools["development"]]
            or identity.get("calibration_source_ids") != [r["source_id"] for r in pools["calibration"]]
            or identity.get("fit_source_ids") != [r["source_id"] for r in pools["fit"][:768]]
            or screen.digest(identity["fit_source_ids"]) != identity.get("fit_source_ids_sha256")
            or identity.get("updates_per_arm") != screen.UPDATES
            or identity.get("forward_batch_sources") != 1
            or identity.get("author_mel") != json.loads(json.dumps(asdict(AuthorMelConfig())))):
        raise ValueError("The completed screen source, panel, or preflight identity changed")
    if (completion.get("arms") != expected_arms or completion.get("updates_per_arm") != screen.UPDATES
            or completion.get("sources_per_arm") != 768 or completion.get("automatic_promotion") is not False):
        raise ValueError("All four preregistered arms must be complete")
    results = read_json(directory/"results.json")
    if results.get("identity") != identity or [r["arm"] for r in results.get("completed_arms",[])] != expected_arms:
        raise ValueError("Combined results do not authenticate all screen arms")
    calibrations = read_json(directory/"calibration.json")
    files = [args.base_out/"preflight.json", args.base_out/"initial.pt", args.base_out/"channel-selection.json",
             args.manifest, Path(__file__), Path(screen.__file__), Path(__file__).with_name("author_mel.py"),
             directory/"screen-identity.json", directory/"completed.json", directory/"results.json", directory/"calibration.json"]
    for arm, definition, _ in screen.ARMS:
        arm_directory = directory/arm
        payload = load_authenticated_arm(arm_directory,arm,identity)
        if payload["identity"]["coefficients"] != calibrations[definition]["coefficients"]:
            raise ValueError("Arm coefficients differ from the recorded calibration")
        del payload
        saved = read_json(arm_directory/"development-step256.json")
        completed = read_json(arm_directory/"completed.json")
        if saved["aggregate"] != completed["common_quality"] or completed not in results["completed_arms"]:
            raise ValueError("Saved source and combined evaluation reports disagree")
        if [row["source_id"] for row in saved["rows"]] != identity["development_source_ids"]:
            raise ValueError("Saved development source order changed")
        files.extend(arm_directory/name for name in ("final.pt","completed.json","launch.json","development-step256.json"))
    hashes = {str(path):base.sha(path) for path in files}
    return metadata, selection, initial, preflight, manifest, pools, identity, hashes


def quiet_summary(rows):
    samples = sum(r["valid_samples"] for r in rows)
    counts = {name:sum(r["failure_category"]==name for r in rows)
              for name in ("passed","residual_only","amplitude_only","both")}
    def rms(key):
        return math.sqrt(sum(r["valid_samples"]*r[key]**2 for r in rows)/samples) if samples else None
    largest = max(rows,key=lambda r:r["residual_abs_max"]) if rows else None
    return {"quiet_windows":len(rows),"quiet_samples":samples,
            "failed_windows":len(rows)-counts["passed"],"failure_categories":counts,
            "residual_rms":rms("residual_rms"),"residual_rms_double":rms("residual_rms_double"),
            "cached_teacher_rms":rms("teacher_rms"),
            "prediction_rms":rms("student_rms"),"window_residual_dc_rms":rms("residual_mean"),
            "centered_residual_rms":rms("centered_residual_rms"),
            "maximum_absolute_error":None if largest is None else {
                key:largest[key] for key in ("source_id","window_id","residual_abs_max",
                    "maximum_error_source_sample","maximum_error_source_seconds",
                    "maximum_error_scored_sample","context_frames","actual_startup",
                    "maximum_error_signed_residual","maximum_error_prediction","maximum_error_cached_teacher")}}


@torch.no_grad()
def quiet_diagnostics(prediction,target,crop):
    raw = base.quiet_window_metrics(prediction,target,torch.ones_like(target,dtype=torch.bool))
    p,t = prediction.detach().cpu().double(),target.detach().cpu().double()
    rows = []
    for original in raw["windows"]:
        if not original["is_quiet"]: continue
        row = dict(original)
        a,b = row["start_sample"],row["stop_sample"]
        pi,ti = p[...,a:b],t[...,a:b]
        residual = pi-ti
        maximum_index = int(residual.abs().reshape(-1).argmax())
        absolute_sample = crop["start_frame"]*1920+a+maximum_index
        residual_failed = row["residual_rms"] > row["residual_limit"]
        amplitude_failed = row["student_rms"] > row["output_rms_limit"]
        row.update({"source_id":crop["source_id"],
            "window_id":crop["source_id"]+":"+str(crop["start_frame"]*1920+a),
            "source_start_sample":crop["start_frame"]*1920+a,
            "source_stop_sample":crop["start_frame"]*1920+b,
            "failure_category":("both" if residual_failed and amplitude_failed else
                "residual_only" if residual_failed else "amplitude_only" if amplitude_failed else "passed"),
            "prediction_mean":float(pi.mean()),"cached_teacher_mean":float(ti.mean()),
            "residual_mean":float(residual.mean()),
            "residual_rms_double":float(residual.square().mean().sqrt()),
            "centered_residual_rms":float((residual-residual.mean()).square().mean().sqrt()),
            "residual_abs_max":float(residual.abs().max()),
            "maximum_error_source_sample":absolute_sample,"maximum_error_source_seconds":absolute_sample/48000,
            "maximum_error_scored_sample":a+maximum_index,
            "context_frames":crop["context_frames"],
            "actual_startup":crop["context_frames"]==0 and crop["start_frame"]==0,
            "maximum_error_signed_residual":float(residual.reshape(-1)[maximum_index]),
            "maximum_error_prediction":float(pi.reshape(-1)[maximum_index]),
            "maximum_error_cached_teacher":float(ti.reshape(-1)[maximum_index])})
        rows.append(row)
    return {"config":raw["config"],"aggregate":quiet_summary(rows),"windows":rows,
            "alignment":"20ms grid of the scored crop, matching original common evaluation; source sample offsets included",
            "limits":"Existing provisional engineering checks, not calibrated audibility thresholds",
            "interpretation":"DC and centered energy decomposition; non-DC residual alone does not distinguish noise from phase error"}


@torch.no_grad()
def fit_quiet_occupancy(crops):
    """Count actual scored target windows, without any encoder/decoder calls."""
    if not crops or len({c["source_id"] for c in crops}) != len(crops):
        raise ValueError("A nonempty unique fitting panel is required")
    rows = []
    for crop in crops:
        start = crop["context_frames"]*1920
        stop = start+crop["valid_scored_samples"]
        target = crop["teacher_audio"]
        if start < 0 or stop > target.shape[-1] or stop <= start:
            raise ValueError("Invalid fitting target extent")
        t = target[...,start:stop]
        measured = base.quiet_window_metrics(t,t,torch.ones_like(t,dtype=torch.bool))
        quiet = [w for w in measured["windows"] if w["is_quiet"]]
        quiet_samples = sum(w["valid_samples"] for w in quiet)
        rows.append({"source_id":crop["source_id"],"samples":t.numel(),
            "quiet_samples":quiet_samples,"active_samples":t.numel()-quiet_samples,
            "windows":measured["window_count"],"quiet_windows":len(quiet),
            "active_windows":measured["window_count"]-len(quiet)})
    aggregate = {key:sum(r[key] for r in rows) for key in
                 ("samples","quiet_samples","active_samples","windows","quiet_windows","active_windows")}
    aggregate.update({"sources":len(rows),"sources_with_quiet":sum(r["quiet_samples"]>0 for r in rows),
                      "quiet_sample_fraction":aggregate["quiet_samples"]/aggregate["samples"]})
    return {"aggregate":aggregate,"rows":rows,"config":measured["config"],
            "source_ids_sha256":screen.digest([r["source_id"] for r in rows]),
            "measurement":"Cached teacher only, same scored-crop20ms grid and partial tails as development checks; no model forward"}


@torch.no_grad()
def evaluate_reference(model, teacher, crops, author, current):
    if not crops or len({r["source_id"] for r in crops}) != len(crops):
        raise ValueError("A nonempty, source-unique development panel is required")
    author_terms, current_terms, rows, quiet_student, quiet_teacher = [], [], [], [], []
    for crop in crops:
        z, target, valid, spans = base.batch([crop])
        trace = base.teacher_forward(teacher,z)
        h = model.group_from_input(trace["group_input"])
        prediction = model.suffix_from_group(h)
        if prediction.shape != target.shape:
            raise ValueError("Prediction and original cached teacher waveform differ in shape")
        start, stop = spans[0]
        expected_valid = torch.zeros_like(valid)
        expected_valid[...,start:stop] = True
        if not torch.equal(valid,expected_valid):
            raise ValueError("The scored contiguous span does not match its mask")
        p, t = prediction[...,start:stop], target[...,start:stop]
        live_teacher = trace["waveform"][...,start:stop]
        student_quiet = quiet_diagnostics(p,t,crop)
        teacher_quiet = quiet_diagnostics(live_teacher,t,crop)
        quiet_student.extend(student_quiet["windows"])
        quiet_teacher.extend(teacher_quiet["windows"])
        a, c = author.group_terms(p,t), current.group_terms(p,t)
        author_terms.append(a); current_terms.append(c)
        ar, cr = author.aggregate(a), current.aggregate([c])
        rows.append({"source_id":crop["source_id"],"samples":p.numel(),
            "author_mel":float(ar.losses["teacher_mel"]),
            "author_log_sums": [float(x) for x in a.log_sums],
            "author_mel_elements":list(a.mel_element_counts),
            "common_mel":float(cr.losses["teacher_mel"]),
            "common_mel_linear":float(cr.losses["teacher_mel_linear"]),
            "common_mel_log":float(cr.losses["teacher_mel_log"]),
            "common_mel_elements":list(c.mel_element_counts),
            "quiet_student":student_quiet["aggregate"],"quiet_teacher_replay":teacher_quiet["aggregate"],
            "teacher_cache_max_absolute_error":float((live_teacher-t).abs().max())})
    a, c = author.aggregate(author_terms), current.aggregate(current_terms)
    return {"aggregate":{"sources":len(rows),"samples":sum(r["samples"] for r in rows),
        "author_mel":float(a.losses["teacher_mel"]),
        "common_mel":float(c.losses["teacher_mel"]),
        "common_mel_linear":float(c.losses["teacher_mel_linear"]),
        "common_mel_log":float(c.losses["teacher_mel_log"])},
        "author_mel_elements":list(a.counts["mel_elements_by_resolution"]),
        "common_mel_elements":list(c.counts["mel_elements_by_resolution"]),"rows":rows,
        "quiet_diagnostics":{"config":student_quiet["config"],
            "student":{"aggregate":quiet_summary(quiet_student),"windows":quiet_student},
            "teacher_replay":{"aggregate":quiet_summary(quiet_teacher),"windows":quiet_teacher},
            "reference":"Both predictions are compared against the same frozen cached teacher waveform",
            "alignment":student_quiet["alignment"],"limits":student_quiet["limits"],
            "interpretation":student_quiet["interpretation"]},
        "reduction":{"author":"Pool actual valid mel elements per resolution, then sum seven log10 means",
                     "common":"Unchanged five-scale linear and natural-log means, pooled before averaging scales"}}


def verify_common(report,saved):
    """Reproduction is an identity check, separate from quality acceptance."""
    pairs = [(report["aggregate"],saved["aggregate"],"aggregate")]
    if [r["source_id"] for r in report["rows"]] != [r["source_id"] for r in saved["rows"]]:
        raise ValueError("Replayed development source order differs")
    if report["common_mel_elements"] != saved["mel_elements"]:
        raise ValueError("Replayed pooled spectral denominators differ")
    for got,old in zip(report["rows"],saved["rows"]):
        if got["samples"] != old["samples"] or got["common_mel_elements"] != old["mel_elements"]:
            raise ValueError("Replayed source sample or spectral counts differ")
        pairs.append((got,old,got["source_id"]))
    differences = []
    for got,old,label in pairs:
        for key in ("mel","mel_linear","mel_log"):
            a,b = got["common_"+key],old[key]
            if not math.isfinite(a) or not math.isclose(a,b,rel_tol=COMMON_RTOL,abs_tol=COMMON_ATOL):
                raise RuntimeError(f"Saved common spectral score was not reproduced: {label}/{key}: {a} versus {b}")
            differences.append(abs(a-b))
    for key in ("sources","samples"):
        if report["aggregate"][key] != saved["aggregate"][key]:
            raise ValueError("Replayed aggregate source or sample counts differ")
    quiet = report["quiet_diagnostics"]["student"]["aggregate"]
    if (quiet["quiet_windows"] != saved["aggregate"]["quiet_windows"]
            or quiet["failed_windows"] != saved["aggregate"]["quiet_failed_windows"]):
        raise ValueError("Replayed quiet window or failure counts differ")
    before,after = saved["aggregate"]["quiet_residual_rms_mean"],quiet["residual_rms"]
    if (before is None) != (after is None) or (before is not None and not math.isclose(before,after,rel_tol=COMMON_RTOL,abs_tol=1e-10)):
        raise RuntimeError("Replayed quiet residual differs")
    return {"passed":True,"atol":COMMON_ATOL,"rtol":COMMON_RTOL,
            "compared_values":len(differences),"maximum_absolute_difference":max(differences)}


def main():
    parser = argparse.ArgumentParser()
    for name in ("assets","base-out","manifest","screen-out","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError("Use a new addendum path; existing reports are immutable")
    base.policy()
    metadata,selection,initial,preflight,manifest,pools,identity,hashes = authenticate_screen(args)
    teacher = base.FrozenAudioVAE2.from_files(args.assets/"audio_vae_v2.py",args.assets/"audiovae.pth",device="cuda")
    model = base.build_student(teacher.model.decoder,selection["stage2_indices"],selection["stage3_indices"])
    model.load_group_state_dict(initial["group"])
    if screen.group_digest(model) != identity["initial_group_sha256"]:
        raise ValueError("Reconstructed initial group differs from the paired screen")
    frozen = screen.frozen_versions(model)
    author,current = AuthorMelLoss().cuda(),base.objective()
    if json.loads(json.dumps(asdict(current.config))) != identity["current_mel"]:
        raise ValueError("The common spectral configuration changed")
    fit_occupancy = fit_quiet_occupancy(pools["fit"][:768])
    if fit_occupancy["source_ids_sha256"] != identity["fit_source_ids_sha256"]:
        raise ValueError("Fitting quiet occupancy uses different sources")
    arms = []
    with torch.no_grad():
        for arm,definition,rate in screen.ARMS:
            payload = load_authenticated_arm(args.screen_out/arm,arm,identity)
            model.load_group_state_dict(payload["group"])
            del payload
            digest_before = screen.group_digest(model)
            report = evaluate_reference(model,teacher,pools["development"],author,current)
            report["common_reproduction"] = verify_common(report,read_json(args.screen_out/arm/"development-step256.json"))
            if screen.group_digest(model) != digest_before or screen.frozen_versions(model) != frozen:
                raise RuntimeError("Read-only scoring changed decoder state")
            report.update({"arm":arm,"training_mel":definition,"learning_rate":rate,
                           "checkpoint_sha256":hashes[str(args.screen_out/arm/"final.pt")],
                           "group_sha256":digest_before})
            arms.append(report)
            base.event("reference_mel_scored",arm=arm,**report["aggregate"])
    for path,checksum in hashes.items():
        if base.sha(path) != checksum: raise RuntimeError("Authenticated input changed while scoring: "+path)
    base.write_json(args.out,{"version":VERSION,"complete":True,"passed":True,
        "original_preflight":metadata,"screen_identity_sha256":screen.digest(identity),
        "input_sha256":hashes,"source_ids":identity["development_source_ids"],
        "fitting_quiet_occupancy":fit_occupancy,
        "author_metric":author.provenance,"arms":arms,"automatic_promotion":False,
        "training_updates":0,"optimizer_created":False,"forward_batch_sources":1,
        "torch":str(torch.__version__),"cudnn":torch.backends.cudnn.version(),
        "interpretation":"Additional metric on the same held-out development panel; not an independent final test set or a new training arm",
        "quality_acceptance":"No new acceptance threshold or ranking; passed means authentication and common-score reproduction passed"})


if __name__ == "__main__": main()
