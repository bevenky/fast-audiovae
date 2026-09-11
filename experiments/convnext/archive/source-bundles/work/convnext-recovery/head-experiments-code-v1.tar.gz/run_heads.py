"""Bounded frozen-body head calibration, teacher migration and heldout screen."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import importlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.quiet_audio import QuietAudioConfig
from audiovae_student.restart_data import file_sha
from audiovae_student.training import _rng_state, _restore_rng
from bounded_head import bounded_waveform, capture_teacher_pre_tanh, transient_errors
from canonical_evaluation import build_canonical_evaluation
from fit_support import (select_training_sources, complete_scored_frames, ReadoutStatistics,
                         validation_masks, select_regularization)
from head_calibration import minimum_norm_anchor, covariance_ridge_fit, SHRINKAGE_GRID
from native_chain import QUARTER_SHA, restore_quarter
from run_comparison import heldout_hashes
from run_update_experiment import load_canonical_panel
from training_overlay import load_training_overlay, verify_overlay_files


def transform(x, mode):
    if mode == "raw": return x
    if mode == "clamp": return bounded_waveform(x)
    if mode == "tanh": return torch.tanh(x)
    raise ValueError(mode)


@dataclass(frozen=True)
class HeadContract:
    name: str
    mode: str
    def to_dict(self): return {"experiment": "head_calibration_v1", "name": self.name, "output_mode": self.mode}


class OutputView(nn.Module):
    def __init__(self, base, name, mode):
        super().__init__(); self.base = base; self.config = base.config
        self.fusion_config = HeadContract(name, mode)
    def forward(self, *args, **kwargs):
        return transform(self.base(*args, **kwargs), self.fusion_config.mode)
    def initial_state(self, *args, **kwargs): return self.base.initial_state(*args, **kwargs)
    def forward_stream(self, *args, **kwargs):
        audio, state = self.base.forward_stream(*args, **kwargs)
        return transform(audio, self.fusion_config.mode), state


@torch.no_grad()
def features(model, z):
    captured = []
    handle = model.output.register_forward_pre_hook(lambda m, a: captured.append(a[0].detach()))
    try: audio = model(z)
    finally: handle.remove()
    if len(captured) != 1: raise RuntimeError("Expected one readout call")
    h = captured[0]
    if not torch.equal(model._waveform(F.conv1d(h, model.output.weight)), audio):
        raise RuntimeError("Observed feature/readout replay differs")
    return h, audio


@torch.no_grad()
def fresh_silence(teacher, model):
    device = teacher.device
    audio = torch.zeros(1, 1, 128000, device=device, dtype=torch.float32)
    with torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=True, allow_tf32=False):
        z = teacher.encode(audio)
    if z.shape != (1, 64, 200): raise RuntimeError("Unexpected eight-second encoded silence")
    values = [teacher.decode(z) for _ in range(3)]
    if not torch.equal(values[1], values[2]): raise RuntimeError("Teacher warmup not stable")
    captured = []
    handle = teacher.model.decoder.model[-2].register_forward_hook(lambda m,a,v: captured.append(v.detach()))
    try: target = teacher.decode(z)
    finally: handle.remove()
    if len(captured)!=1 or not torch.equal(target, values[2]) or not torch.equal(torch.tanh(captured[0]), target):
        raise RuntimeError("Teacher pre-tanh path mismatch")
    h, _ = features(model, z)
    hh = h[0, :, 400:].double().reshape(2048, 100, 4)
    yy = target[0, 0, 192000:].double().reshape(100, 4, 480)
    pp = captured[0][0, 0, 192000:].double().reshape(100, 4, 480)
    if any(float((v-v.select(dim,0).unsqueeze(dim)).abs().max()) != 0 for v,dim in ((hh,1),(yy,0),(pp,0))):
        raise RuntimeError("Fresh silence is not stationary over its final four seconds")
    return {"H": hh.mean(1).cpu(), "post": yy.mean(0).T.cpu(), "pre": pp.mean(0).T.cpu(),
        "receipt": {"input_samples16k": 128000, "latent_shape": list(z.shape),
            "input_sha256": state_fingerprint(audio), "latents_sha256": state_fingerprint(z),
            "teacher_waveform_sha256": state_fingerprint(target), "teacher_pre_tanh_sha256": state_fingerprint(captured[0]),
            "calibration_seconds": [4,8], "cycle_variation_max": 0.,
            "encoder_policy": "Singleton raw mu, scoped cuDNN disabled, matching canonical target contract",
            "interpretation": "Synthetic calibration family, not a heldout generalization claim"}}


@torch.no_grad()
def validation_score(weight, mode, bank, device):
    weight = weight.to(device)[...,None]
    sums = {"square":0., "absolute":0., "quiet_square":0., "valid_samples":0, "quiet_samples":0}
    for row in bank:
        p = F.conv1d(row["h"].to(device), weight).transpose(1,2).reshape(1,1,-1)
        p = transform(p, mode)
        t,v,q = [row[k].to(device) for k in ("target","valid","quiet")]
        e = p.double()-t.double()
        sums["square"] += float(e[v].square().sum()); sums["absolute"] += float(e[v].abs().sum())
        sums["quiet_square"] += float(e[q].square().sum())
        sums["valid_samples"] += int(v.sum()); sums["quiet_samples"] += int(q.sum())
    if min(sums["valid_samples"],sums["quiet_samples"])<=0: raise RuntimeError("Validation needs natural quiet and scored samples")
    return {"mse":sums["square"]/sums["valid_samples"],"mae":sums["absolute"]/sums["valid_samples"],
        "quiet_mse":sums["quiet_square"]/sums["quiet_samples"],
        "valid_samples":sums["valid_samples"],"quiet_samples":sums["quiet_samples"]}


def compare(control, candidate):
    if control["evaluation_contract_sha256"]!=candidate["evaluation_contract_sha256"]:
        raise RuntimeError("Quality evaluation contracts differ")
    a,b=control["recovery_metrics"],candidate["recovery_metrics"]
    def ratio(x,y): return x/y if y else (1. if x==0 else None)
    checks=[]
    def details(report, group):
        pairs=[(r,t) for r,t in zip(report["rows"],report["transient_diagnostics"],strict=True)
            if not r["source_id"].startswith("encoded_") and
               (group=="natural" or "cohort/"+r["cohort"]==group)]
        n=sum(r["high_frequency"]["elements"] for r,t in pairs)
        result={"hf_magnitude_mae":sum(r["high_frequency"]["magnitude_mae"]*r["high_frequency"]["elements"] for r,t in pairs)/n,
            "hf_complex_rms":math.sqrt(sum(r["high_frequency"]["complex_residual_rms"]**2*r["high_frequency"]["elements"] for r,t in pairs)/n)}
        for metric in ("first_difference","second_difference"):
            result[metric+"_rms"]=math.sqrt(sum(t[metric]["squared_error_sum"] for r,t in pairs)/sum(t[metric]["count"] for r,t in pairs))
        return result
    for group,limit in [("natural",1.01),("cohort/speech",1.02),("cohort/expressive",1.02)]:
        for metric in ("raw_mae","mel"):
            r=ratio(b[group][metric],a[group][metric]);checks.append({"name":group+"/"+metric,"ratio":r,"limit":limit,"passed":r is not None and r<=limit})
        reference,observed=details(control,group),details(candidate,group)
        for metric in reference:
            r=ratio(observed[metric],reference[metric]);checks.append({"name":group+"/"+metric,"ratio":r,"limit":limit,"passed":r is not None and r<=limit})
    for group in a:
        if group.startswith(("language/","condition/")) and a[group]["sources"]>=2:
            for metric in ("raw_mae","mel"):
                r=ratio(b[group][metric],a[group][metric]);checks.append({"name":group+"/"+metric,"ratio":r,"limit":1.05,"passed":r is not None and r<=1.05})
    quiet_ratio=ratio(b["natural"]["quiet_residual_rms"],a["natural"]["quiet_residual_rms"])
    steady_ratio=ratio(candidate["encoded_zero_steady_2_to_6_seconds"]["quiet_residual_rms"],control["encoded_zero_steady_2_to_6_seconds"]["quiet_residual_rms"])
    return {"natural_mae_ratio":ratio(b["natural"]["raw_mae"],a["natural"]["raw_mae"]),
        "natural_mel_ratio":ratio(b["natural"]["mel"],a["natural"]["mel"]),
        "natural_quiet_ratio":quiet_ratio,"steady_silence_ratio":steady_ratio,
        "maximum_peak":b["all"]["maximum_peak"],"overshoot_observations":b["all"]["scored_overshoot_samples"],
        "quality_screen_passed":all(c["passed"] for c in checks),"checks":checks,
        "all_three_targets_passed":all(c["passed"] for c in checks) and quiet_ratio is not None and quiet_ratio<=.90 and steady_ratio is not None and steady_ratio<1 and b["all"]["maximum_peak"]<=1,
        "promotion":False,"meaning":"Engineering reconstruction screen; not perceptual equivalence or production approval"}


@torch.no_grad()
def run(args):
    from diagnostic_common import load_context, atomic_json, status
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(args.checkpoint).resolve(strict=True)
    if file_sha(checkpoint)!=QUARTER_SHA:raise ValueError("Wrong baseline checkpoint")
    ctx=load_context()
    panel=load_canonical_panel(ctx,{"requires_canonical_evaluation":True},args.canonical_receipt)
    overlay=load_training_overlay(ctx,args.training_receipt,required_counts={"targeted_generator":12800})
    hashes,inventory=heldout_hashes(panel,args.canonical_inventory)
    pool=overlay["pools"]["targeted_generator"]
    selection=select_training_sources(pool,ctx.data["rows"],{c.source_id for c in panel["crops"]},hashes)
    atomic_json(out/"selection.json",selection)
    saved=torch.load(checkpoint,map_location="cpu",mmap=True,weights_only=True)
    engine=ctx.engine("targeted",device="cuda");original=restore_quarter(engine,saved["engine"])
    base=engine.model;original_weight=base.output.weight.detach().clone();w=original_weight[...,0].cpu()
    rng=_rng_state();modes={m:m.training for m in base.modules()};base.eval()
    teacher=ctx.teacher();teacher.model.eval();teacher_sha=ctx.parent["identity"]["data"]["teacher_state_sha256"]
    config=engine.config.quiet_audio
    if isinstance(config,dict):config=QuietAudioConfig(**config)
    sources={Path(__file__).resolve()}
    for module in ("head_calibration","bounded_head","fit_support","diagnostic_common","native_chain","canonical_evaluation","run_comparison","training_overlay"):
        sources.add(Path(importlib.import_module(module).__file__).resolve())
    identity={"version":"head_calibration_v1","checkpoint_sha256":QUARTER_SHA,"restored_engine_sha256":original,
        "canonical_panel":panel["identity"],"training_overlay":overlay["identity"],"inventory":inventory,
        "selection_sha256":selection["identity_sha256"],"teacher_state_sha256":teacher_sha,
        "torch":str(torch.__version__),"cudnn":torch.backends.cudnn.version(),"ridge_grid":list(SHRINKAGE_GRID),
        "sources":{str(p):file_sha(p) for p in sources},"fit_count":2048,"validation_count":256,
        "policy":"Frozen body; training-only calibration selection. No GAN, optimizer update, automatic retention or production change."}
    atomic_json(out/"identity.json",identity)
    started=time.monotonic()
    try:
        anchor=fresh_silence(teacher,base);atomic_json(out/"silence-calibration.json",anchor["receipt"])
        stats={name:ReadoutStatistics(original_weight) for name in ("post","pre")}
        counts=[]
        for count,index in enumerate(selection["splits"]["fit"]["indices"],1):
            crop=pool[index];h,raw=features(base,crop.latents.to(engine.device))
            capture=capture_teacher_pre_tanh(teacher,crop,config)
            hh,tt,receipt=complete_scored_frames(crop,h)
            hp,tp,_=complete_scored_frames(crop,h,target_override=capture["pre_tanh"])
            stats["post"].add(hh,tt,source_id=crop.source_id);stats["pre"].add(hp,tp,source_id=crop.source_id)
            counts.append({**receipt,"teacher_capture":capture["receipt"]})
            if count%64==0:status("head_fit_features",completed=count,total=2048)
        moments={name:value.finalize() for name,value in stats.items()}
        atomic_json(out/"fit-accounting.json",{"crops":counts,"summary":{k:{a:b for a,b in v.items() if a not in ("A","B")} for k,v in moments.items()}})
        bank=[]
        for count,index in enumerate(selection["splits"]["validation"]["indices"],1):
            crop=pool[index];h,raw=features(base,crop.latents.to(engine.device))
            target,valid,quiet=validation_masks(crop,device=engine.device,quiet_config=config)
            bank.append({"h":h.cpu(),"target":target.cpu(),"valid":valid.cpu(),"quiet":quiet.cpu()})
        baseline_val=validation_score(w,"raw",bank,engine.device)
        weights={"baseline":w};fit_reports={};candidates=[("baseline","raw"),("baseline","clamp")]
        for mode in ("shared","full"):
            result=minimum_norm_anchor(w,anchor["H"],anchor["post"],mode=mode)
            name=mode+"_anchor";weights[name]=result.weight;fit_reports[name]=result.report
            candidates.extend(((name,"raw"),(name,"clamp")))
        for family,output_mode in (("post","raw"),("pre","tanh")):
            trials=[];fits={}
            for shrinkage in SHRINKAGE_GRID:
                status("head_ridge_solve",family=family,shrinkage=shrinkage)
                v=moments[family]
                result=covariance_ridge_fit(w,anchor["H"],anchor[family],v["A"],v["B"],shrinkage=shrinkage)
                metrics=validation_score(result.weight,output_mode,bank,engine.device)
                trials.append({"lambda":shrinkage,**metrics});fits[shrinkage]=result
            choice=select_regularization(trials,baseline_val)
            atomic_json(out/(family+"-selection.json"),choice)
            fit_reports[family+"_ridge_trials"]={str(k):v.report for k,v in fits.items()}
            if choice["selected_lambda"] is not None:
                result=fits[choice["selected_lambda"]];name=family+"_ridge"
                weights[name]=result.weight;candidates.append((name,output_mode))
                if family=="post":candidates.append((name,"clamp"))
        atomic_json(out/"calibration-results.json",{"fits":fit_reports,"candidates":candidates,"baseline_training_validation":baseline_val})
        torch.save({"format_version":1,"base_checkpoint_sha256":QUARTER_SHA,"head_weights":weights,
                    "candidates":candidates,"qualification":"Experimental final-readout weights only; upstream model stays pinned and frozen"},out/"heads.pt")
        reports={};comparisons={}
        for name,mode in candidates:
            key=name+"_"+mode;status("head_canonical_evaluation",candidate=key)
            base.output.weight.copy_(weights[name].to(engine.device)[...,None])
            view=OutputView(base,name,mode);engine.model=view
            transient=[]
            def observe(module,inputs,prediction):
                crop=panel["crops"][len(transient)]
                target,valid,_=validation_masks(crop,device=engine.device,quiet_config=config)
                transient.append({"source_id":crop.source_id,"start_frame":crop.start_frame,**transient_errors(prediction,target,valid)})
            handle=view.register_forward_hook(observe)
            try:report=build_canonical_evaluation(engine,panel["crops"],panel["metadata"],panel["receipt"])
            finally:handle.remove();engine.model=base
            report["transient_diagnostics"]=transient
            atomic_json(out/(key+".json"),report)
            reports[key]=report
            comparisons[key]=compare(reports["baseline_raw"],report)
            atomic_json(out/"comparisons.json",comparisons)
        base.output.weight.copy_(original_weight)
        for m,mode in modes.items():m.training=mode
        _restore_rng(rng)
        if state_fingerprint(engine.state_dict())!=original:raise RuntimeError("Retained engine state changed")
        if state_fingerprint(teacher.model.state_dict())!=teacher_sha:raise RuntimeError("Frozen teacher changed")
        if file_sha(checkpoint)!=QUARTER_SHA:raise RuntimeError("Retained checkpoint changed")
        for key in ("receipt","cache"):
            if file_sha(panel["identity"][key+"_path"])!=panel["identity"][key+"_sha256"]:raise RuntimeError("Heldout input changed")
        atomic_json(out/"complete.json",{"complete":True,"seconds":time.monotonic()-started,"engine_state_preserved":True,
            "original_files":ctx.verify_files(),"fresh_files":verify_overlay_files(overlay),"comparisons":comparisons,
            "head_file_sha256":file_sha(out/"heads.pt"),"promoted":False})
        status("head_experiments_complete",out=str(out))
    finally:
        engine.model=base;base.output.weight.copy_(original_weight)
        for m,mode in modes.items():m.training=mode
        _restore_rng(rng)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint","training-receipt","canonical-receipt","canonical-inventory","out"):p.add_argument("--"+arg,required=True)
    args=p.parse_args()
    lock=Path("/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)
