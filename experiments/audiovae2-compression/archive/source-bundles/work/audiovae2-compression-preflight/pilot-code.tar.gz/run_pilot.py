"""Bounded whole-group distillation with pinned cached teacher waveforms.

The preflight is separate from training. It records initialization, numerical
checks and calibrated coefficients before any persistent fitting update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from audiovae_student.teacher import FrozenAudioVAE2, SOURCE_SHA256, CHECKPOINT_SHA256
from audiovae_student.reconstruction_v2 import ReconstructionV2, ReconstructionV2Config
from audiovae_student.quiet_audio import quiet_window_metrics
from group_model import build_student, teacher_trace
from monitoring import LOSS_SPECIFICATION, log_training, log_validation, log_boundaries
from boundary_diagnostics import evaluate_boundaries


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def event(stage, **values):
    print(json.dumps({"stage": stage, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values}, allow_nan=False), flush=True)


def policy():
    torch.set_num_threads(1)
    torch.manual_seed(20260910)
    np.random.seed(20260910)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_data(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    stable = {k:v for k,v in manifest.items() if k != "identity_sha256"}
    checksum = hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if checksum != manifest["identity_sha256"]:
        raise ValueError("Manifest checksum differs")
    if sha(manifest["overlay_receipt_path"]) != manifest["overlay_receipt_sha256"]:
        raise ValueError("Overlay receipt differs from selection manifest")
    receipt = json.loads(Path(manifest["overlay_receipt_path"]).read_text())
    cache = Path(receipt["path"])
    if str(cache) != manifest["cache_path"] or receipt["sha256"] != manifest["cache_sha256"] or sha(cache) != receipt["sha256"]:
        raise ValueError("Cached teacher pair bytes changed")
    saved = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    if saved["contract"] != receipt["contract"]:
        raise ValueError("Cache receipt and tensor contract disagree")
    source_pools = saved["pools"]
    pools = {}
    for name in ("calibration", "development", "fit"):
        pools[name] = []
        items = manifest["splits"][name]["rows"]
        if [r["pool_index"] for r in items] != manifest["splits"][name]["indices"]:
            raise ValueError("Selection indices and rows disagree")
        for item in items:
            crop = source_pools[manifest["pool"]][item["pool_index"]]
            if crop["source_id"] != item["source_id"] or crop["start_frame"] != item["start_frame"]:
                raise ValueError("Manifest source/crop index mismatch")
            if crop["context_start_frame"] and crop["context_frames"] < 20:
                raise ValueError("Insufficient original teacher causal history")
            if crop["valid_scored_samples"] < 4096:
                raise ValueError("Pilot reconstruction requires a complete largest spectral window")
            pools[name].append(crop)
        if len({c["source_id"] for c in pools[name]}) != len(pools[name]):
            raise ValueError("Source reuse inside a pilot pool")
    for a, b in (("calibration", "development"), ("fit", "development"), ("fit", "calibration")):
        if {c["source_id"] for c in pools[a]} & {c["source_id"] for c in pools[b]}:
            raise ValueError("Pilot pools overlap")
    return manifest, pools, receipt


def batch(crops, device="cuda"):
    if not crops:
        raise ValueError("Cannot construct an empty batch")
    for crop in crops:
        zc, tc = crop["latents"], crop["teacher_audio"]
        if zc.ndim != 3 or zc.shape[:2] != (1,64) or tc.ndim != 3 or tc.shape[:2] != (1,1):
            raise ValueError("Invalid cached latent/waveform shape")
        if tc.shape[-1] != zc.shape[-1]*1920 or zc.dtype != torch.float32 or tc.dtype != torch.float32:
            raise ValueError("Invalid cached dtype or waveform duration")
        context, count = crop["context_frames"], crop["valid_scored_samples"]
        if context < 0 or count <= 0 or context*1920+count > tc.shape[-1]:
            raise ValueError("Scored samples exceed their own crop, before batch padding")
        if (crop["context_start_frame"] < 0 or crop["start_frame"]-crop["context_start_frame"] != context):
            raise ValueError("Inconsistent absolute causal context")
    frames = max(c["latents"].shape[-1] for c in crops)
    z = torch.cat([F.pad(c["latents"], (0, frames-c["latents"].shape[-1])) for c in crops]).to(device)
    t = torch.cat([F.pad(c["teacher_audio"], (0, frames*1920-c["teacher_audio"].shape[-1])) for c in crops]).to(device)
    valid = torch.zeros_like(t, dtype=torch.bool)
    spans = []
    for i, c in enumerate(crops):
        start = c["context_frames"] * 1920
        stop = start + c["valid_scored_samples"]
        if stop > t.shape[-1]:
            raise ValueError("Invalid scored crop extent")
        valid[i, :, start:stop] = True
        spans.append((start, stop))
    return z, t, valid, spans


@torch.no_grad()
def teacher_forward(teacher, z):
    # Runtime initialization belongs to each fresh process and input shape.
    # Targets retain FP32 and the validated singleton-cache provenance.
    key = (tuple(z.shape), str(z.dtype), str(z.device))
    warmed = getattr(teacher, "_compression_warmed_shapes", set())
    if key not in warmed:
        for _ in range(3):
            teacher_trace(teacher.model.decoder, z)
        warmed.add(key)
        teacher._compression_warmed_shapes = warmed
    return teacher_trace(teacher.model.decoder, z)


def objective():
    return ReconstructionV2(ReconstructionV2Config(
        fft_sizes=(256, 512, 1024, 2048, 4096),
        mel_bands=(32, 64, 64, 128, 128))).cuda()


def reconstruction_denominators(crops, spectral):
    """Exact corpus counts without opening audio or constructing target graphs."""
    sizes, bands = spectral.config.fft_sizes, spectral.config.mel_bands
    lengths = [c["valid_scored_samples"] for c in crops]
    if not lengths or any(type(n) is not int or n < max(sizes) for n in lengths):
        raise ValueError("Every scored crop must accommodate the largest spectral window")
    return {"samples": sum(lengths), "mel_elements": tuple(
        sum(((n-size)//(size//4)+1)*band for n in lengths) for size,band in zip(sizes,bands))}


def _reconstruction(terms, spectral, denominators=None):
    if denominators is None:
        return spectral.aggregate(terms).losses
    sample_count = denominators["samples"]
    counts = denominators["mel_elements"]
    if sample_count < sum(t.sample_count for t in terms) or len(counts) != len(spectral.config.fft_sizes):
        raise ValueError("Invalid pooled reconstruction denominators")
    if any(c <= 0 or c < sum(t.mel_element_counts[i] for t in terms) for i,c in enumerate(counts)):
        raise ValueError("Invalid pooled spectral denominators")
    waveform = sum(t.waveform_absolute_sum for t in terms) / sample_count
    linear = torch.stack([sum(t.mel_linear_sums[i] for t in terms)/c for i,c in enumerate(counts)]).mean()
    logarithmic = torch.stack([sum(t.mel_log_sums[i] for t in terms)/c for i,c in enumerate(counts)]).mean()
    return {"teacher_waveform": waveform,
            "teacher_mel": spectral.config.mel_linear_weight*linear + spectral.config.mel_log_weight*logarithmic,
            "teacher_mel_linear": linear, "teacher_mel_log": logarithmic}


def losses(p, target, h, ht, valid, spans, spectral, denominators=None):
    if p.shape != target.shape or h.shape != ht.shape or h.shape[-1]*4 != p.shape[-1]:
        raise ValueError("Waveform and whole-group boundaries are misaligned")
    if valid.shape != p.shape or valid.dtype != torch.bool or len(spans) != p.shape[0]:
        raise ValueError("Scored mask and spans must match waveform geometry")
    expected = torch.zeros_like(valid)
    for i,(a,b) in enumerate(spans):
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= p.shape[-1]:
            raise ValueError("Invalid scored span")
        expected[i,:,a:b] = True
    if not torch.equal(expected, valid):
        raise ValueError("Waveform spans and feature validity mask disagree")
    terms = [spectral.group_terms(p[i:i+1, :, a:b], target[i:i+1, :, a:b]) for i, (a,b) in enumerate(spans)]
    rec = _reconstruction(terms, spectral, denominators)
    weights = valid.reshape(valid.shape[0], 1, h.shape[-1], 4).sum(-1).to(h)
    selected = weights > 0
    prediction = h.masked_fill(~selected, 0)
    reference = ht.detach().masked_fill(~selected, 0)
    sample_count = denominators["samples"] if denominators is not None else weights.sum()
    feature = ((prediction-reference).square()*weights).sum() / (sample_count*h.shape[1])
    return {"waveform": rec["teacher_waveform"], "mel": rec["teacher_mel"], "feature": feature}


def grad_norm(grads):
    return math.sqrt(sum(float(g.double().square().sum()) for g in grads))


def parameters(model):
    return [p for _, p in model.group_named_parameters()]


def pivoted_subset(gram, rank):
    """Coordinate selection by deterministic pivoted Cholesky of X.T @ X."""
    if (gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not torch.isfinite(gram).all()
            or type(rank) is not int or not 1 <= rank <= gram.shape[0]):
        raise ValueError("Need a finite square Gram matrix and a valid positive integer rank")
    g = gram.detach().cpu().double()
    n = g.shape[0]
    remaining = torch.ones(n, dtype=torch.bool)
    diagonal = g.diag().clone().clamp_min_(0)
    factors = torch.zeros(n, rank, dtype=torch.float64)
    chosen = []
    for k in range(rank):
        score = diagonal.masked_fill(~remaining, -1)
        index = int(score.argmax())
        chosen.append(index)
        remaining[index] = False
        d = float(diagonal[index])
        if d <= torch.finfo(torch.float64).eps * max(1.0, float(g.diag().max())):
            continue
        column = g[:, index] - factors[:, :k] @ factors[index, :k]
        factors[:, k] = column / math.sqrt(d)
        diagonal.sub_(factors[:, k].square()).clamp_min_(0)
    return sorted(chosen)


@torch.no_grad()
def choose_channels(teacher, crops):
    grams = {"stage2_output": torch.zeros(512,512,device="cuda",dtype=torch.float64),
             "stage3_output": torch.zeros(256,256,device="cuda",dtype=torch.float64)}
    for number, crop in enumerate(crops):
        z, _, _, spans = batch([crop])
        trace = teacher_forward(teacher, z)
        for name, rate in (("stage2_output",1200), ("stage3_output",6000)):
            a,b = spans[0]
            value = trace[name][0, :, a//(48000//rate):b//(48000//rate)]
            gen = torch.Generator().manual_seed(20260910+number)
            indices = torch.randperm(value.shape[-1], generator=gen)[:min(256,value.shape[-1])].to(value.device)
            sample = value[:,indices].double()
            grams[name].add_(sample @ sample.T)
    return {"stage2_indices": pivoted_subset(grams["stage2_output"],256),
            "stage3_indices": pivoted_subset(grams["stage3_output"],128),
            "method": "Uncentered activation Gram, deterministic coordinate pivoting; maximum256 selected scored time cells per source"}


def assert_close(a,b,label,atol=1e-5,rtol=1e-4):
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError(label+": invalid shape or values")
    error = float((a-b).abs().max())
    if not torch.allclose(a,b,atol=atol,rtol=rtol):
        raise RuntimeError(label+": maximum error "+str(error))
    return error


@torch.no_grad()
def numerical_preflight(teacher, crops):
    copy_model = build_student(teacher.model.decoder, list(range(512)), list(range(256)))
    records = []
    for offset,crop in enumerate(crops):
        z,t,valid,_ = batch([crop])
        trace = teacher_forward(teacher,z)
        p,h = copy_model.forward_latents(z)
        repeated = teacher_forward(teacher,z)
        row = {"sources":[crop["source_id"]], "execution_batch_size":1,
               "copy_wave_max_abs":assert_close(p,trace["waveform"],"copy waveform"),
               "copy_boundary_max_abs":assert_close(h,trace["group_output"],"copy boundary"),
               "sealed_wave_max_abs":assert_close(trace["waveform"][valid],t[valid],"sealed teacher targets"),
               "singleton_repeat_max_abs":{key:assert_close(trace[key],repeated[key],"teacher repeat "+key)
                                           for key in ("group_input","group_output","waveform")}}
        records.append(row)
        event("copy_parity",completed=offset+1,total=len(crops))
    del copy_model
    torch.cuda.empty_cache()
    return records


@torch.no_grad()
def compressed_batch_check(model, teacher, crops):
    records=[]
    for offset in range(0,len(crops),3):
        chosen=crops[offset:offset+3]
        z,_,_,_=batch(chosen)
        x=teacher_forward(teacher,z)["group_input"]
        for _ in range(3):
            h=model.group_from_input(x);model.suffix_from_group(h)
        h=model.group_from_input(x);p=model.suffix_from_group(h)
        for i,crop in enumerate(chosen):
            length=crop["latents"].shape[-1]*8
            xi=x[i:i+1,:,:length]
            for _ in range(3):
                hi=model.group_from_input(xi);model.suffix_from_group(hi)
            hi=model.group_from_input(xi);pi=model.suffix_from_group(hi)
            records.append({"source_id":crop["source_id"],
                "group_max_abs":assert_close(h[i:i+1,:,:hi.shape[-1]],hi,"compressed batch group"),
                "waveform_max_abs":assert_close(p[i:i+1,:,:pi.shape[-1]],pi,"compressed batch waveform")})
    return {"passed":True,"records":records}


@torch.no_grad()
def singleton_forward_check(model, teacher, crops):
    """Check the same B1 computation in deployed and training/evaluation roles."""
    records=[]
    for crop in crops:
        z,_,_,_=batch([crop])
        tr=teacher_forward(teacher,z)
        for _ in range(3):
            model.forward_latents(z)
            model.suffix_from_group(model.group_from_input(tr["group_input"]))
        full_p,full_h=model.forward_latents(z)
        h=model.group_from_input(tr["group_input"]);p=model.suffix_from_group(h)
        repeated_h=model.group_from_input(tr["group_input"])
        repeated_p=model.suffix_from_group(repeated_h)
        records.append({"source_id":crop["source_id"], "execution_batch_size":1,
            "full_split_group_max_abs":assert_close(full_h,h,"singleton full/split group"),
            "full_split_waveform_max_abs":assert_close(full_p,p,"singleton full/split waveform"),
            "repeated_group_max_abs":assert_close(h,repeated_h,"singleton repeated group"),
            "repeated_waveform_max_abs":assert_close(p,repeated_p,"singleton repeated waveform")})
    return {"passed":True,"execution_batch_size":1,"records":records}


def feature_statistics(prediction,target,valid):
    if (prediction.shape != target.shape or valid.dtype != torch.bool
            or valid.shape != (prediction.shape[0],1,prediction.shape[-1]*4)):
        raise ValueError("Feature diagnostics and waveform mask are misaligned")
    weights=valid.reshape(valid.shape[0],1,prediction.shape[-1],4).sum(-1).double()
    selected=weights>0
    p=prediction.detach().double().masked_fill(~selected,0)
    t=target.detach().double().masked_fill(~selected,0)
    count=int(valid.sum())*prediction.shape[1]
    if not count:raise ValueError("Feature diagnostics require valid samples")
    error_sum=float(((p-t).square()*weights).sum())
    target_sum=float((t.square()*weights).sum())
    prediction_sum=float((p.square()*weights).sum())
    dot=float((p*t*weights).sum())
    power=target_sum/count
    denominator=math.sqrt(target_sum*prediction_sum)
    return {"group_elements":count,"group_squared_error_sum":error_sum,
            "group_teacher_square_sum":target_sum,"group_prediction_square_sum":prediction_sum,"group_dot_sum":dot,
            "group_mse":error_sum/count,"group_nrmse":math.sqrt(error_sum/count/max(power,1e-8)),
            "group_nrmse_floor_applied":power<1e-8,"group_teacher_mean_square":power,
            "group_cosine":dot/denominator if denominator else None}


def select_boundary_panel(crops,source_metadata):
    """A fixed twelve-source diagnostic subset selected without model outputs."""
    indic={"as","bn","brx","doi","gu","hi","kn","ks","kok","mai","ml",
           "mni","mr","ne","or","pa","sa","sat","sd","ta","te","ur"}
    international={"en","es","pt","cmn","ja","fr","ar"}
    by_id={crop["source_id"]:crop for crop in crops}
    if len(by_id)!=len(crops) or any(key not in source_metadata for key in by_id):
        raise ValueError("Boundary panel requires distinct development sources and complete metadata")
    ordered=sorted(by_id,key=lambda key:hashlib.sha256(("boundary-panel-v1|"+key).encode()).hexdigest())
    selected=[];reasons=[];used=set()
    def take(reason,predicate):
        key=next((key for key in ordered if key not in used and predicate(source_metadata[key])),None)
        if key is None:raise ValueError("Insufficient development coverage for boundary panel: "+reason)
        used.add(key);selected.append(by_id[key]);reasons.append({"source_id":key,"reason":reason})
        return source_metadata[key]
    event_groups={"whistling":{"human_whistling_source_description"},
                  "loud_vocalization":{"Screaming","Shout","Yell"},
                  "laughter":{"Laughter","Giggle","Chuckle_and_chortle"},
                  "crying_or_whisper":{"Crying_and_sobbing","Whispering","explicit_whisper_style","Breathing"}}
    for name,labels in event_groups.items():
        take("expressive/"+name,lambda row,labels=labels:bool(labels & set(row.get("verified_source_labels",[]))))
    for kind in ("quiet","transition"):
        for _ in range(2):take(kind,lambda row,kind=kind:row.get("selection_kind")==kind)
    for name,languages in (("indic",indic),("international",international)):
        represented=set()
        for _ in range(2):
            row=take(name,lambda row:row.get("normalized_language") in languages-represented)
            represented.add(row["normalized_language"])
    return selected,{"version":"boundary-panel-v1","sources":reasons,
                     "selection":"Fixed source metadata only, without loss or model-output ranking",
                     "coverage_limit":"Twelve diagnostic sources; expressive labels are recording-level, not timestamped event occupancy"}


@torch.no_grad()
def evaluate(model, teacher, crops, spectral, batch_size=1):
    if not crops or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Evaluation requires scored crops and a positive batch size")
    rows=[]
    spectral_terms=[]
    for offset in range(0,len(crops),batch_size):
        chosen=crops[offset:offset+batch_size]
        z,t,valid,spans=batch(chosen)
        tr=teacher_forward(teacher,z)
        h=model.group_from_input(tr["group_input"])
        p=model.suffix_from_group(h)
        for i,crop in enumerate(chosen):
            a,b=spans[i]
            pi,ti=p[i:i+1,:,a:b],t[i:i+1,:,a:b]
            e=(pi-ti).double();tt=ti.double();pp=pi.double()
            q=quiet_window_metrics(pi,ti,torch.ones_like(pi,dtype=torch.bool))
            quiet=[w for w in q["windows"] if w["is_quiet"]]
            n=pi.numel();nr=sum(w["valid_samples"] for w in quiet)
            term=spectral.group_terms(pi,ti)
            spectral_terms.append(term)
            rec=spectral.aggregate([term]).losses
            rms=float(tt.square().mean().sqrt())
            active=torch.zeros_like(pi,dtype=torch.bool)
            for window in q["windows"]:
                if not window["is_quiet"]:
                    active[...,window["start_sample"]:window["stop_sample"]]=True
            active_p,active_t=pp[active],tt[active]
            den=float(active_p.norm()*active_t.norm())
            mse=float(e.square().mean())
            group=feature_statistics(h[i:i+1],tr["group_output"][i:i+1],valid[i:i+1])
            rows.append({"source_id":crop["source_id"],"samples":n,
                "mae":float(e.abs().mean()),"mse":mse,
                "waveform_residual_rms":math.sqrt(mse),"waveform_nrmse":math.sqrt(mse/max(rms*rms,1e-8)),
                "waveform_nrmse_floor_applied":rms*rms<1e-8,**group,
                "mel":float(rec["teacher_mel"]),"mel_linear":float(rec["teacher_mel_linear"]),
                "mel_log":float(rec["teacher_mel_log"]),"mel_elements":list(term.mel_element_counts),"teacher_rms":rms,
                "nonquiet_samples":int(active.sum()),
                "cosine":float((active_p*active_t).sum())/den if den else None,
                "quiet_samples":nr,"quiet_error_sum":sum(w["valid_samples"]*w["residual_rms"]**2 for w in quiet),
                "quiet_windows":len(quiet),"quiet_failed":q["quiet_failed_count"],
                "peak":float(pi.abs().max()),"overshoot":int((pi.abs()>1+2e-6).sum())})
    n=sum(r["samples"] for r in rows);qn=sum(r["quiet_samples"] for r in rows)
    cos=[r["cosine"] for r in rows if r["cosine"] is not None]
    pooled=spectral.aggregate(spectral_terms)
    group_elements=sum(r["group_elements"] for r in rows)
    group_error=sum(r["group_squared_error_sum"] for r in rows)
    group_target=sum(r["group_teacher_square_sum"] for r in rows)
    group_prediction=sum(r["group_prediction_square_sum"] for r in rows)
    group_dot=sum(r["group_dot_sum"] for r in rows)
    group_power=group_target/group_elements
    group_den=math.sqrt(group_target*group_prediction)
    waveform_mse=sum(r["mse"]*r["samples"] for r in rows)/n
    waveform_power=sum(r["teacher_rms"]**2*r["samples"] for r in rows)/n
    aggregate={"sources":len(rows),"samples":n,
        **{k:sum(r[k]*r["samples"] for r in rows)/n for k in ("mae","mse")},
        "mel":float(pooled.losses["teacher_mel"]),
        "mel_linear":float(pooled.losses["teacher_mel_linear"]),"mel_log":float(pooled.losses["teacher_mel_log"]),
        "nonquiet_samples":sum(r["nonquiet_samples"] for r in rows),
        "waveform_residual_rms":math.sqrt(waveform_mse),
        "waveform_nrmse":math.sqrt(waveform_mse/max(waveform_power,1e-8)),
        "waveform_nrmse_floor_applied":waveform_power<1e-8,
        "group_mse":group_error/group_elements,
        "group_nrmse":math.sqrt(group_error/group_elements/max(group_power,1e-8)),
        "group_nrmse_floor_applied":group_power<1e-8,
        "group_cosine":group_dot/group_den if group_den else None,
        "nonquiet_cosine_mean":sum(cos)/len(cos) if cos else None,
        "quiet_residual_rms_mean":math.sqrt(sum(r["quiet_error_sum"] for r in rows)/qn) if qn else None,
        "quiet_windows":sum(r["quiet_windows"] for r in rows),"quiet_failed_windows":sum(r["quiet_failed"] for r in rows),
        "peak_abs_max":max(r["peak"] for r in rows),"overshoot_samples":sum(r["overshoot"] for r in rows)}
    if aggregate["overshoot_samples"]:
        raise RuntimeError("Bounded teacher head produced overshoots")
    return {"aggregate":aggregate,"rows":rows,"mel_elements":list(pooled.counts["mel_elements_by_resolution"]),
            "reduction":{"waveform":"valid samples pooled","mel":"valid elements pooled per resolution, then resolutions averaged",
                         "nonquiet_cosine":"equal-source mean over only active teacher-conditioned20ms windows",
                         "group":"all channels pooled with exact valid waveform-sample weights, including partial cells",
                         "nrmse":"root mean-square error divided by teacher RMS, teacher mean-square floor1e-8 explicitly flagged",
                         "quiet_residual":"RMS of sample-pooled quiet squared residual"}}


def calibration(model,teacher,crops,spectral,out):
    params=parameters(model)
    groups=[[crop] for crop in crops]
    denominators=reconstruction_denominators(crops,spectral)
    sums={k:[torch.zeros_like(p) for p in params] for k in ("waveform","mel","feature")}
    means={k:0.0 for k in sums}
    original={k:v.detach().cpu().clone() for k,v in model.group_state_dict().items()}
    original_grads=[None if p.grad is None else p.grad.detach().clone() for p in params]
    cpu_rng=torch.get_rng_state();numpy_rng=np.random.get_state()
    cuda_rng=torch.cuda.get_rng_state_all() if any(p.is_cuda for p in params) else None
    probes=[]
    try:
        for chosen in groups:
            z,t,valid,spans=batch(chosen)
            with torch.no_grad():tr=teacher_forward(teacher,z)
            h=model.group_from_input(tr["group_input"]);p=model.suffix_from_group(h)
            branches=losses(p,t,h,tr["group_output"],valid,spans,spectral,denominators)
            for k,value in branches.items():
                means[k]+=float(value.detach())
                grad=torch.autograd.grad(value,params,retain_graph=k!="feature")
                for total,g in zip(sums[k],grad):total.add_(g.detach())
        norms={k:grad_norm(v) for k,v in sums.items()}
        if any(not math.isfinite(v) or v<=0 for v in norms.values()):
            raise RuntimeError("Undefined compressed-model gradient calibration")
        coeff={"waveform":1.0,"mel":0.5*norms["waveform"]/norms["mel"],"feature":0.5*norms["waveform"]/norms["feature"]}
        for rate in (1e-4,3e-5,1e-5):
            model.load_group_state_dict(original)
            opt=torch.optim.AdamW(params,lr=rate,betas=(.9,.99),weight_decay=0,eps=1e-8)
            for i,param in enumerate(params):param.grad=sum(coeff[k]*sums[k][i] for k in coeff)
            opt.step();opt.zero_grad(set_to_none=True)
            if any(not torch.isfinite(p).all() for p in params):
                raise RuntimeError("Nonfinite disposable calibration update")
            predicted={k:sum(float((g.double()*(p.detach()-original[name].to(p)).double()).sum())
                             for g,(name,p) in zip(sums[k],model.group_named_parameters())) for k in coeff}
            after={k:0.0 for k in coeff}
            with torch.no_grad():
                for chosen in groups:
                    z,t,valid,spans=batch(chosen);tr=teacher_forward(teacher,z)
                    h=model.group_from_input(tr["group_input"]);p=model.suffix_from_group(h)
                    for k,v in losses(p,t,h,tr["group_output"],valid,spans,spectral,denominators).items():after[k]+=float(v)
            passed=all(math.isfinite(after[k]) and after[k]<=means[k]+1e-6*max(means[k],1e-8) for k in coeff)
            probes.append({"learning_rate":rate,"after":after,"predicted_changes":predicted,
                           "actual_changes":{k:after[k]-means[k] for k in coeff},"all_branches_nonincreasing":passed})
            if passed:break
    finally:
        model.load_group_state_dict(original)
        for p,g in zip(params,original_grads):p.grad=g
        torch.set_rng_state(cpu_rng);np.random.set_state(numpy_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
    report={"before":means,"pooled_parameter_gradient_norms":norms,"coefficients":coeff,"optimizer_probes":probes,
        "denominators":denominators,"reduction":"all calibration samples and valid mel elements pooled before gradient summation",
        "state_restored":True,
        "learning_rate":probes[-1]["learning_rate"] if probes[-1]["all_branches_nonincreasing"] else None,
        "optimizer":{"name":"AdamW","betas":[.9,.99],"weight_decay":0,"eps":1e-8},
        "passed":probes[-1]["all_branches_nonincreasing"]}
    write_json(out/"calibration.json",report)
    if not report["passed"]:raise RuntimeError("No calibrated optimizer probe improved all teacher objectives")
    return report


def training_update(model,teacher,crops,spectral,coefficients,optimizer,*,record_diagnostics=False):
    """One optimizer step from singleton forwards, pooled over the chosen sources."""
    denominators=reconstruction_denominators(crops,spectral)
    accumulated={key:0.0 for key in ("waveform","mel","feature")}
    optimizer.zero_grad(set_to_none=True)
    for crop in crops:
        z,t,valid,spans=batch([crop])
        with torch.no_grad():tr=teacher_forward(teacher,z)
        h=model.group_from_input(tr["group_input"]);p=model.suffix_from_group(h)
        branches=losses(p,t,h,tr["group_output"],valid,spans,spectral,denominators)
        total=sum(coefficients[key]*value for key,value in branches.items())
        if not torch.isfinite(total):raise RuntimeError("Nonfinite training objective")
        total.backward()
        for key,value in branches.items():accumulated[key]+=float(value.detach())
    params=parameters(model)
    if any(param.grad is None for param in params):raise RuntimeError("Missing group gradient")
    if not torch.stack([torch.isfinite(param.grad).all() for param in params]).all():
        raise RuntimeError("Nonfinite group gradient")
    diagnostics={}
    if record_diagnostics:
        diagnostics["gradient_norm"]=float(torch.linalg.vector_norm(torch.stack([param.grad.detach().norm() for param in params])))
        if params[0].is_cuda:diagnostics["gpu_memory_gib"]=torch.cuda.memory_allocated(params[0].device)/(1024**3)
    optimizer.step()
    if record_diagnostics and params[0].is_cuda:torch.cuda.synchronize(params[0].device)
    return {"total":sum(coefficients[key]*value for key,value in accumulated.items()),**accumulated,**diagnostics}


def checkpoint(path,model,opt,step,identity,extra=None):
    state={k:v.detach().cpu() for k,v in model.group_state_dict().items()}
    value={"format":"audiovae2_group_width_v1","group":state,"step":step,"identity":identity,
           "optimizer":None if opt is None else opt.state_dict(),"torch_rng":torch.get_rng_state(),
           "cuda_rng":torch.cuda.get_rng_state_all(),"extra":extra}
    tmp=path.with_suffix(".tmp")
    torch.save(value,tmp)
    with tmp.open("rb") as f:os.fsync(f.fileno())
    os.replace(tmp,path)


def authenticate_streaming_check(report,expected):
    if (report.get("version")!="audiovae2_group_streaming_v1" or report.get("passed") is not True
            or any(report.get(key)!=value for key,value in expected.items())):
        raise ValueError("Streaming check belongs to different weights, source or preflight")
    if report.get("device")!="cpu" or report.get("threads")!=1 or report.get("chunk_sizes_latent_frames")!=[1,2,4]:
        raise ValueError("Streaming check did not verify the expected CPU chunk sizes")
    before=report.get("teacher_state_sha256_before")
    if not before or before!=report.get("teacher_state_sha256_after"):
        raise ValueError("Streaming check did not preserve the frozen teacher")
    models=report.get("models",{})
    if set(models)!={"full_width_control","narrowed_initial"}:
        raise ValueError("Streaming check must cover both the copied control and narrowed candidate")
    for name,model in models.items():
        before=model.get("model_state_sha256_before")
        if model.get("passed") is not True or not before or before!=model.get("model_state_sha256_after"):
            raise ValueError("Streaming model parity or state preservation failed: "+name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("preflight","train"))
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--assets",type=Path,required=True)
    ap.add_argument("--tensorboard",type=Path)
    ap.add_argument("--steps",type=int,default=1000)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    policy()
    manifest,pools,receipt=load_data(args.manifest)
    source_metadata={row["source_id"]:row for row in manifest["splits"]["development"]["rows"]}
    boundary_panel,boundary_selection=select_boundary_panel(pools["development"],source_metadata)
    teacher=FrozenAudioVAE2.from_files(args.assets/"audio_vae_v2.py",args.assets/"audiovae.pth",device="cuda")
    spectral=objective()
    identity={"manifest_sha256":sha(args.manifest),"cache_sha256":receipt["sha256"],
        "teacher_source_sha256":SOURCE_SHA256,"teacher_checkpoint_sha256":CHECKPOINT_SHA256,
        "runner_sha256":sha(__file__),"model_sha256":sha(Path(__file__).with_name("group_model.py")),
        "monitoring_sha256":sha(Path(__file__).with_name("monitoring.py")),
        "boundary_diagnostics_sha256":sha(Path(__file__).with_name("boundary_diagnostics.py")),
        "boundary_panel":boundary_selection,
        "torch":str(torch.__version__),"cudnn":torch.backends.cudnn.version(),
        "precision":"FP32 TF32 disabled","batch_size":3,
        "execution_batch_size":1,"gradient_accumulation_steps":3,
        "shared_helpers_sha256":{name:sha(sys.modules["audiovae_student."+name].__file__) for name in
            ("teacher","reconstruction_v2","quiet_audio","losses_distillation")},
        "fit_sources":len(pools["fit"]),"scored_fit_hours":sum(c["valid_scored_samples"] for c in pools["fit"])/48000/3600}
    if args.mode=="preflight":
        if (args.out/"initial.pt").exists():raise FileExistsError("Do not overwrite an initialized experiment")
        write_json(args.out/"training-configuration.json",LOSS_SPECIFICATION)
        write_json(args.out/"boundary-panel.json",boundary_selection)
        event("preflight_start",**identity)
        checks=numerical_preflight(teacher,pools["calibration"]+pools["development"])
        write_json(args.out/"copy-parity.json",{"passed":True,"records":checks})
        selection=choose_channels(teacher,pools["calibration"])
        write_json(args.out/"channel-selection.json",selection)
        model=build_student(teacher.model.decoder,selection["stage2_indices"],selection["stage3_indices"])
        write_json(args.out/"singleton-forward-parity.json",singleton_forward_check(model,teacher,pools["calibration"]))
        cal=calibration(model,teacher,pools["calibration"],spectral,args.out)
        before=evaluate(model,teacher,pools["development"],spectral)
        write_json(args.out/"development-step0.json",before)
        boundary_report=evaluate_boundaries(teacher,model,boundary_panel,selection,batch,teacher_forward)
        write_json(args.out/"boundaries-step0.json",boundary_report)
        checkpoint(args.out/"initial.pt",model,None,0,identity)
        write_json(args.out/"preflight.json",{"passed":True,"identity":identity,"calibration":cal,
            "initial_sha256":sha(args.out/"initial.pt"),"selection_sha256":sha(args.out/"channel-selection.json"),
            "development":before["aggregate"],"export_verified":False})
        event("preflight_complete",**before["aggregate"])
        return
    pre=json.loads((args.out/"preflight.json").read_text())
    if not pre["passed"] or pre["identity"]!=identity or sha(args.out/"initial.pt")!=pre["initial_sha256"]:
        raise ValueError("Preflight identity changed")
    if not (args.out/"export-check.json").exists():
        raise ValueError("Export correctness must pass before the pilot")
    export=json.loads((args.out/"export-check.json").read_text())
    expected_export={"preflight_sha256":sha(args.out/"preflight.json"),
        "initial_sha256":pre["initial_sha256"],"selection_sha256":pre["selection_sha256"],
        "model_sha256":identity["model_sha256"],"teacher_source_sha256":SOURCE_SHA256,
        "teacher_checkpoint_sha256":CHECKPOINT_SHA256}
    if not export["passed"] or any(export.get(k)!=v for k,v in expected_export.items()):
        raise ValueError("Export check belongs to different weights, source or preflight")
    if sha(export["onnx_path"])!=export["onnx_sha256"]:
        raise ValueError("Verified ONNX artifact bytes changed")
    if not (args.out/"streaming-check.json").exists():
        raise ValueError("Streaming correctness must pass before the pilot")
    streaming=json.loads((args.out/"streaming-check.json").read_text())
    authenticate_streaming_check(streaming,{**expected_export,
        "streaming_checker_sha256":sha(Path(__file__).with_name("streaming_preflight.py")),
        "manifest_sha256":identity["manifest_sha256"],"cache_sha256":identity["cache_sha256"],
        "runner_sha256":identity["runner_sha256"]})
    selection=json.loads((args.out/"channel-selection.json").read_text())
    if sha(args.out/"channel-selection.json")!=pre["selection_sha256"]:raise ValueError("Selection changed")
    model=build_student(teacher.model.decoder,selection["stage2_indices"],selection["stage3_indices"])
    initial=torch.load(args.out/"initial.pt",weights_only=True,map_location="cpu")
    model.load_group_state_dict(initial["group"])
    cal=pre["calibration"];coeff=cal["coefficients"]
    opt=torch.optim.AdamW(parameters(model),lr=cal["learning_rate"],betas=(.9,.99),weight_decay=0,eps=1e-8)
    if args.steps<=0 or args.steps*3>len(pools["fit"]):raise ValueError("Pilot would repeat sources")
    if (args.out/"train.jsonl").exists():raise FileExistsError("Use an explicit resume to protect existing pilot history")
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(args.tensorboard or args.out/"tensorboard"))
    step0=json.loads((args.out/"development-step0.json").read_text())
    write_json(args.out/"validation-group-summaries-step0.json",log_validation(writer,step0,0,source_metadata))
    log_boundaries(writer,json.loads((args.out/"boundaries-step0.json").read_text()),0)
    writer.flush()
    start=time.monotonic();seen=set();exposure=0
    write_json(args.out/"launch.json",{"identity":identity,"steps":args.steps,"calibration":cal,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    for step in range(1,args.steps+1):
        step_start=time.monotonic()
        chosen=pools["fit"][(step-1)*3:step*3]
        for c in chosen:
            if c["source_id"] in seen:raise RuntimeError("Repeated fitting source")
            seen.add(c["source_id"]);exposure+=c["valid_scored_samples"]/48000
        record_diagnostics=step==1 or step%25==0
        branches=training_update(model,teacher,chosen,spectral,coeff,opt,record_diagnostics=record_diagnostics)
        record={"step":step,**branches,
            "unique_sources":len(seen),"audio_hours":exposure/3600,"elapsed_seconds":time.monotonic()-start}
        if record_diagnostics:record["step_seconds"]=time.monotonic()-step_start
        with (args.out/"train.jsonl").open("a") as f:f.write(json.dumps(record,allow_nan=False)+"\n")
        log_training(writer,record,coeff,cal["learning_rate"],step)
        if step==1 or step%25==0:event("training",**record);writer.flush()
        if step in (256,args.steps):
            report=evaluate(model,teacher,pools["development"],spectral)
            write_json(args.out/f"development-step{step}.json",report)
            write_json(args.out/f"validation-group-summaries-step{step}.json",log_validation(writer,report,step,source_metadata))
            boundary_report=evaluate_boundaries(teacher,model,boundary_panel,selection,batch,teacher_forward)
            write_json(args.out/f"boundaries-step{step}.json",boundary_report)
            log_boundaries(writer,boundary_report,step)
            event("development",step=step,**report["aggregate"])
        if step%100==0 or step in (256,args.steps):
            checkpoint(args.out/"latest.pt",model,opt,step,identity,{"sources_seen":sorted(seen),"audio_hours":exposure/3600})
    writer.flush();writer.close()
    write_json(args.out/"completed.json",{"step":args.steps,"identity":identity,"unique_sources":len(seen),"audio_hours":exposure/3600,"checkpoint_sha256":sha(args.out/"latest.pt")})
    event("pilot_completed",step=args.steps,unique_sources=len(seen))


if __name__=="__main__":main()
