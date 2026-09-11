"""Bounded B1 accumulation versus B3 student-only batching diagnostic.

The authenticated teacher is always called at its original singleton shape.
No optimizer is created and no parameter update is performed. This diagnostic
does not change the training entry point or relax its output parity tolerance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

import settings_screen as screen

base = screen.base
VERSION = "audiovae2_effective_batch3_check_v1"
OUTPUT_ATOL, OUTPUT_RTOL = 1e-5, 1e-4
GRAD_ATOL, GRAD_RTOL, GRAD_RELATIVE_L2 = 1e-7, 1e-3, 1e-3


def tensor_comparison(reference, candidate, atol, rtol):
    if reference.shape != candidate.shape:
        raise ValueError("Compared tensors have different shapes")
    a, b = reference.detach().double(), candidate.detach().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"passed": False, "finite": False}
    difference = b-a
    an, bn, en = float(a.norm()), float(b.norm()), float(difference.norm())
    return {"passed": bool(torch.allclose(a,b,atol=atol,rtol=rtol)), "finite": True,
            "maximum_absolute_error": float(difference.abs().max()),
            "reference_l2": an, "difference_l2": en,
            "relative_l2": en/an if an else (0. if en == 0 else None),
            "cosine": float((a*b).sum())/(an*bn) if an and bn else None,
            "atol": atol, "rtol": rtol}


def padded_references(references):
    """Pad frozen singleton features only after teacher inference is complete."""
    length = max(row["x"].shape[-1] for row in references)
    result = {}
    for key, multiplier in (("x",1),("ht",60),("target",240),("valid",240)):
        result[key] = torch.cat([F.pad(row[key],(0,length*multiplier-row[key].shape[-1]))
                                 for row in references])
    result["spans"] = [row["spans"][0] for row in references]
    return result


def make_reference(teacher, crop, check=False):
    z,target,valid,spans = base.batch([crop])
    if z.shape[0] != 1:
        raise RuntimeError("Teacher batching is prohibited by this diagnostic")
    trace = base.teacher_forward(teacher,z)
    if check:
        base.assert_close(trace["waveform"][valid],target[valid],"sealed teacher target")
    return {"x":trace["group_input"].detach(),"ht":trace["group_output"].detach(),
            "target":target,"valid":valid,"spans":spans}


def student_pass(model, references, spectral, denominators, coefficients, physical_batch,
                 backward=True, capture=False):
    groups = [[row] for row in references] if physical_batch == 1 else [references]
    values = {key:0. for key in coefficients}
    outputs = []
    for group in groups:
        row = group[0] if len(group) == 1 else padded_references(group)
        h = model.group_from_input(row["x"])
        p = model.suffix_from_group(h)
        branches = base.losses(p,row["target"],h,row["ht"],row["valid"],row["spans"],spectral,denominators)
        if backward:
            sum(coefficients[key]*value for key,value in branches.items()).backward()
        for key,value in branches.items():
            values[key] += float(value.detach())
        if capture:
            for index,source in enumerate(group):
                valid = source["valid"]
                hp = h[index:index+1,:,:source["ht"].shape[-1]]
                pp = p[index:index+1,:,:source["target"].shape[-1]]
                feature_valid = valid.reshape(1,1,hp.shape[-1],4).any(-1).expand_as(hp)
                outputs.append({"waveform":pp.detach()[valid].cpu(),
                                "group":hp.detach()[feature_valid].cpu()})
    return values,outputs


def check_gradients(model, reference):
    rows = []
    energy = error = dot = candidate_energy = 0.
    for name,param in model.group_named_parameters():
        if param.grad is None:
            raise RuntimeError("Missing student parameter gradient: "+name)
        a,b = reference[name],param.grad.detach().cpu()
        comparison = tensor_comparison(a,b,GRAD_ATOL,GRAD_RTOL)
        rows.append({"name":name,**comparison})
        aa,bb = a.double(),b.double()
        energy += float(aa.square().sum()); candidate_energy += float(bb.square().sum())
        error += float((aa-bb).square().sum()); dot += float((aa*bb).sum())
    relative = math.sqrt(error/energy) if energy else (0. if not error else None)
    return {"passed": all(row["passed"] for row in rows) and relative is not None and relative <= GRAD_RELATIVE_L2,
            "relative_l2":relative, "relative_l2_limit":GRAD_RELATIVE_L2,
            "cosine":dot/math.sqrt(energy*candidate_energy) if energy*candidate_energy else None,
            "failed_parameter_count":sum(not row["passed"] for row in rows),
            "parameters":rows}


def select_panels(crops, metadata):
    used=set()
    def take(values):
        chosen=[row for row in values if row["source_id"] not in used][:3]
        if len(chosen)!=3: raise ValueError("Required distinct diagnostic sources unavailable")
        used.update(row["source_id"] for row in chosen)
        return chosen
    same=take(row for row in crops if row["latents"].shape[-1]==94 and row["valid_scored_samples"]==122880)
    expressive=sorted([row for row in crops if metadata[row["source_id"]].get("verified_source_labels")],
                      key=lambda row:row["latents"].shape[-1])
    mixed=take([expressive[0],expressive[len(expressive)//2],expressive[-1]])
    quiet=take(sorted([row for row in crops if metadata[row["source_id"]].get("selection_kind") in ("quiet","transition")],
                     key=lambda row:(row["context_frames"],row["source_id"])))
    return {"equal_length":same,"mixed_expressive_lengths":mixed,"quiet_and_transitions":quiet}


def end_to_end(model,teacher,crops,spectral,coefficients,physical_batch):
    """Match one effective batch of three, omitting only the common Adam step."""
    model.zero_grad(set_to_none=True)
    denominators=base.reconstruction_denominators(crops,spectral)
    if physical_batch==1:
        for crop in crops:
            references=[make_reference(teacher,crop)]
            student_pass(model,references,spectral,denominators,coefficients,1)
    else:
        references=[make_reference(teacher,crop) for crop in crops]
        student_pass(model,references,spectral,denominators,coefficients,3)
    params=base.parameters(model)
    if any(param.grad is None for param in params) or not torch.stack([torch.isfinite(p.grad).all() for p in params]).all():
        raise RuntimeError("Missing or nonfinite benchmark gradient")


def benchmark(model,teacher,crops,spectral,coefficients):
    for batch in (1,3):
        for _ in range(2): end_to_end(model,teacher,crops,spectral,coefficients,batch)
    records=[]
    for repeat in range(4):
        for batch in ((1,3) if repeat%2==0 else (3,1)):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start=time.perf_counter();start_event.record()
            end_to_end(model,teacher,crops,spectral,coefficients,batch)
            end_event.record();torch.cuda.synchronize()
            records.append({"repeat":repeat,"physical_batch":batch,"wall_seconds":time.perf_counter()-start,
                            "cuda_elapsed_seconds":start_event.elapsed_time(end_event)/1000,
                            "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3})
    medians={str(batch):statistics.median(r["wall_seconds"] for r in records if r["physical_batch"]==batch) for batch in (1,3)}
    return {"records":records,"median_wall_seconds":medians,"speedup_b3":medians["1"]/medians["3"],
            "condition":"Three original singleton teacher forwards plus student forward/backward, transfer, validation and scalar extraction; no optimizer update or disk/data loading. Two warmups then four alternating paired measurements.",
            "effective_sources":3,"scored_audio_seconds":sum(row["valid_scored_samples"] for row in crops)/48000}


def main():
    parser=argparse.ArgumentParser()
    for name in ("base-out","manifest","assets","checkpoint","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError("Use a new diagnostic report path")
    base.policy()
    metadata,selection,initial,preflight,manifest,pools=screen.authenticate_inputs(args)
    receipt=json.loads(args.checkpoint.with_suffix(".json").read_text())
    checksum=base.sha(args.checkpoint)
    if receipt["checkpoint_sha256"]!=checksum or receipt["step"]!=1000 or not receipt["frozen_state_preserved"]:
        raise ValueError("Expected authenticated completed1000 checkpoint")
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if payload["step"]!=1000 or payload["identity"]["channel_selection"]!=selection or payload["identity"]["definition"]!="current":
        raise ValueError("Checkpoint identity differs from selected current-loss continuation")
    teacher=base.FrozenAudioVAE2.from_files(args.assets/"audio_vae_v2.py",args.assets/"audiovae.pth",device="cuda")
    model=base.build_student(teacher.model.decoder,selection["stage2_indices"],selection["stage3_indices"])
    model.load_group_state_dict(payload["group"])
    original_digest=screen.group_digest(model)
    frozen=screen.frozen_versions(model)
    teacher_before={name:(p.data_ptr(),p._version) for name,p in teacher.state_dict().items()}
    spectral=base.objective();coefficients=payload["identity"]["coefficients"]
    metadata_rows={row["source_id"]:row for row in manifest["splits"]["development"]["rows"]}
    panels=select_panels(pools["development"],metadata_rows)
    report={"version":VERSION,"script_sha256":base.sha(__file__),"checkpoint_sha256":checksum,
            "original_preflight":metadata,"teacher_policy":"Original singleton-only teacher inference; concatenate/pad detached outputs after inference",
            "output_tolerance":{"atol":OUTPUT_ATOL,"rtol":OUTPUT_RTOL},"effective_batch":3,
            "optimizer_created":False,"parameter_updates":0,"automatic_promotion":False,
            "torch":str(torch.__version__),"cudnn":torch.backends.cudnn.version(),
            "device":torch.cuda.get_device_name(),"precision":"FP32, TF32 disabled, deterministic algorithms",
            "panels":[]}
    for panel_name,crops in panels.items():
        base.event("batch_diagnostic_panel",panel=panel_name,sources=[c["source_id"] for c in crops])
        references=[make_reference(teacher,crop,check=True) for crop in crops]
        denominators=base.reconstruction_denominators(crops,spectral)
        # Warm each student's actual shape before saving the singleton reference.
        with torch.no_grad():
            for physical in (1,3):
                for _ in range(3):student_pass(model,references,spectral,denominators,coefficients,physical,backward=False)
        model.zero_grad(set_to_none=True)
        single_values,single_outputs=student_pass(model,references,spectral,denominators,coefficients,1,capture=True)
        single_gradients={name:p.grad.detach().cpu().clone() for name,p in model.group_named_parameters()}
        model.zero_grad(set_to_none=True)
        batched_values,batched_outputs=student_pass(model,references,spectral,denominators,coefficients,3,capture=True)
        comparisons=[]
        for crop,a,b in zip(crops,single_outputs,batched_outputs):
            comparisons.append({"source_id":crop["source_id"],"latent_frames":crop["latents"].shape[-1],
                                "valid_scored_samples":crop["valid_scored_samples"],
                                **{key:tensor_comparison(a[key],b[key],OUTPUT_ATOL,OUTPUT_RTOL) for key in ("waveform","group")}})
        loss_comparison={key:{"singleton":single_values[key],"batched":batched_values[key],
                             "passed":math.isclose(single_values[key],batched_values[key],rel_tol=1e-4,abs_tol=1e-7)} for key in coefficients}
        gradients=check_gradients(model,single_gradients)
        passed=all(row[key]["passed"] for row in comparisons for key in ("waveform","group")) and all(r["passed"] for r in loss_comparison.values()) and gradients["passed"]
        row={"name":panel_name,"sources":comparisons,"losses":loss_comparison,"gradients":gradients,"passed":passed}
        del references,single_outputs,batched_outputs,single_gradients
        if passed:
            row["timing"]=benchmark(model,teacher,crops,spectral,coefficients)
        else:
            row["timing_skipped"]="Strict numerical parity failed; no speed-based selection is justified"
        report["panels"].append(row)
        base.write_json(args.out,report)
        base.event("batch_diagnostic_result",panel=panel_name,passed=passed,
                   failed_gradients=gradients["failed_parameter_count"],gradient_relative_l2=gradients["relative_l2"])
        if not passed:break
    model.zero_grad(set_to_none=True)
    unchanged=(screen.group_digest(model)==original_digest and screen.frozen_versions(model)==frozen
               and teacher_before=={name:(p.data_ptr(),p._version) for name,p in teacher.state_dict().items()}
               and base.sha(args.checkpoint)==checksum)
    if not unchanged:raise RuntimeError("Diagnostic changed teacher, student or saved checkpoint")
    report.update({"complete":True,"state_preserved":True,
                   "passed":len(report["panels"])==len(panels) and all(p["passed"] for p in report["panels"]),
                   "scope":"Nine fixed development sources in three source-disjoint triplets, unless stopped at a strict parity failure. Not a full-corpus equivalence guarantee."})
    base.write_json(args.out,report)


if __name__=="__main__":main()
