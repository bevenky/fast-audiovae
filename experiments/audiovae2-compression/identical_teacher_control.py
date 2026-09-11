"""Post5000 full-width original-teacher identity and training-path control.

The native decoder is reference A; immutable full-source cached audio is B.
Exact equality is reported separately from the existing numerical tolerance.
No original checkpoint, target, optimizer, training runner or event is modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
import group_model as group
import progressive_model as progressive
import progressive_train as prior

base, screen, replay, resume, joint = prior.base, prior.screen, prior.replay, prior.resume, prior.joint
VERSION = 'audiovae2_identical_teacher_control_v1'
TOTAL_SOURCES, STABILITY_UPDATES, ACCUMULATION = 60000, 12, 12
ATOL, RTOL = 1e-5, 1e-4
STAGES = {2:'stage1_output',3:'stage2_output',4:'stage3_output',5:'stage4_output',
          6:'stage5_output',7:'stage6_output',9:'pre_tanh'}
quiet_metrics = base.quiet_window_metrics


def state_hash(module):
    checksum = hashlib.sha256()
    for name, value in module.state_dict().items():
        checksum.update(name.encode()); checksum.update(str(value.dtype).encode())
        checksum.update(str(tuple(value.shape)).encode())
        checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def build_control(independently_loaded_decoder):
    selection = {key:list(range(independently_loaded_decoder.model[index].block[1].out_channels))
                 for key,index in zip(progressive.KEYS,(3,4))}
    return progressive.initialize_from_teacher(independently_loaded_decoder, selection)


def state_identity(teacher_decoder, control):
    a, b = teacher_decoder.state_dict(), control.decoder.state_dict()
    if set(a) != set(b): raise ValueError('Original and control state keys differ')
    mismatches, aliases = [], []
    for key, value in a.items():
        other = b[key]
        if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(value,other):
            mismatches.append(key)
        if value.device == other.device and value.data_ptr() == other.data_ptr(): aliases.append(key)
    return {'equal':not mismatches,'independent_storage':not aliases,'tensors':len(a),
            'mismatches':mismatches,'aliased_tensors':aliases,
            'teacher_state_sha256':state_hash(teacher_decoder),'control_state_sha256':state_hash(control.decoder)}


def compare_tensors(actual, expected, valid=None):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('Comparison tensor shape or dtype differs')
    if valid is None: valid = torch.ones_like(expected,dtype=torch.bool)
    if valid.dtype != torch.bool or valid.shape != expected.shape:
        raise ValueError('Comparison requires an exact boolean sample mask')
    a, b = actual.detach()[valid], expected.detach()[valid]
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite compared tensor')
    n = a.numel()
    if not n:
        return {'elements':0,'exact':True,'allclose_existing':True,'max_abs':0.,'residual_rms':None,
                'mean_residual':None,'expected_rms':None,'actual_rms':None,'atol':ATOL,'rtol':RTOL}
    aa, bb = a.double(), b.double(); delta = aa-bb
    return {'elements':n,'exact':bool(torch.equal(a,b)),
            'allclose_existing':bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL)),
            'max_abs':float(delta.abs().max()),'residual_rms':float(delta.square().mean().sqrt()),
            'mean_residual':float(delta.mean()),'expected_rms':float(bb.square().mean().sqrt()),
            'actual_rms':float(aa.square().mean().sqrt()),'atol':ATOL,'rtol':RTOL}


def cache_check(crop, prediction):
    target = crop['teacher_audio'].to(prediction.device)
    latents = crop['latents']; context = crop['context_frames']; count = crop['valid_scored_samples']
    if (latents.ndim != 3 or latents.shape[:2] != (1,64) or latents.dtype != torch.float32
            or target.ndim != 3 or target.shape[:2] != (1,1)
            or target.shape[-1] != latents.shape[-1]*1920 or target.dtype != torch.float32
            or prediction.dtype != torch.float32 or prediction.shape != target.shape
            or type(context) is not int or context < 0 or type(count) is not int or count <= 0
            or type(crop['start_frame']) is not int or type(crop['context_start_frame']) is not int
            or crop['context_start_frame'] < 0 or crop['start_frame']-crop['context_start_frame'] != context
            or context*1920+count > target.shape[-1]):
        raise ValueError('Cached source/context/scored geometry differs')
    if not torch.isfinite(latents).all() or not torch.isfinite(target).all() or not torch.isfinite(prediction).all():
        raise ValueError('Nonfinite cached or predicted data')
    start, stop = context*1920, context*1920+count
    valid = torch.zeros_like(target,dtype=torch.bool); valid[...,start:stop] = True
    context_mask = torch.zeros_like(valid); context_mask[...,:start] = True
    tail = torch.zeros_like(valid); tail[...,stop:] = True
    return {'source_id':crop['source_id'],'start_frame':crop['start_frame'],
            'context_start_frame':crop['context_start_frame'],'context_frames':context,
            'valid_scored_samples':count,'scored_span':[start,stop],
            'valid':compare_tensors(prediction,target,valid),
            'context':compare_tensors(prediction,target,context_mask),
            'right_padding':compare_tensors(prediction,target,tail),
            'full':compare_tensors(prediction,target)}


def native_decoder_forward(decoder, z, sr_cond=None):
    if sr_cond is None:
        sr_cond = torch.tensor([48000],device=z.device,dtype=torch.int32)
    elif not isinstance(sr_cond,torch.Tensor):
        sr_cond = torch.as_tensor(sr_cond,device=z.device,dtype=torch.int32)
    return decoder(z,sr_cond=sr_cond)


@contextmanager
def capture_boundaries(decoder):
    captured, handles = {}, []
    def hook(key):
        def observe(module, inputs, output): captured[key] = output.detach()
        return observe
    try:
        for index,key in STAGES.items():
            handles.append(decoder.model[index].register_forward_hook(hook(key)))
        if decoder.sr_bin_boundaries is not None:
            for index in range(2,8):
                handles.append(decoder.sr_cond_model[index].register_forward_hook(hook(f'stage{index-1}_conditioned_input')))
        yield captured
    finally:
        for handle in handles: handle.remove()


@torch.no_grad()
def compare_forward(teacher_decoder, control, z, sr_cond=None, *, include_trace=True,
                    return_tensors=False, detailed_boundaries=True):
    if detailed_boundaries:
        with capture_boundaries(teacher_decoder) as native_stages:
            native = native_decoder_forward(teacher_decoder,z,sr_cond)
        with capture_boundaries(control.decoder) as student_stages:
            student = control.forward_from_latents(z,sr_cond)['waveform']
        if set(native_stages) != set(student_stages): raise RuntimeError('Native boundary capture differs')
    else:
        if include_trace: raise ValueError('Trace comparisons require detailed native boundaries')
        native_stages = student_stages = {}
        native = native_decoder_forward(teacher_decoder,z,sr_cond)
        student = control.forward_from_latents(z,sr_cond)['waveform']
    comparisons = {'native_vs_student':compare_tensors(student,native),
                   **{key:compare_tensors(student_stages[key],value) for key,value in native_stages.items()}}
    if include_trace:
        tr = group.teacher_trace(teacher_decoder,z,sr_cond)
        comparisons['native_vs_teacher_trace'] = compare_tensors(tr['waveform'],native)
        for key in STAGES.values():
            comparisons['native_vs_trace/'+key] = compare_tensors(tr[key],native_stages[key])
    result = {'all_exact':all(r['exact'] for r in comparisons.values()),
              'allclose_existing':all(r['allclose_existing'] for r in comparisons.values()),
              'comparisons':comparisons,'student_training':control.training,
              'teacher_training':teacher_decoder.training,'reference':'published native decoder forward'}
    return (result,native,student) if return_tensors else result


def zero_safe_metrics(prediction,target,valid):
    comparison = compare_tensors(prediction,target,valid)
    a, b = prediction.detach()[valid].double(), target.detach()[valid].double()
    denominator = float(a.norm()*b.norm()); power = float(b.square().sum())
    quiet = quiet_metrics(prediction,target,valid)
    return {**comparison,'mae':float((a-b).abs().mean()) if a.numel() else None,
            'cosine':float((a*b).sum())/denominator if denominator else None,
            'rms_ratio':math.sqrt(float(a.square().sum())/power) if power else None,
            'teacher_mean':float(b.mean()) if b.numel() else None,
            'prediction_mean':float(a.mean()) if a.numel() else None,
            'quiet_windows':quiet['quiet_window_count'],'quiet_failed':quiet['quiet_failed_count'],
            'quiet_passed':quiet['quiet_failed_count']==0 if quiet['quiet_window_count'] else None,'quiet':quiet}


def training_forward_probe(model,teacher,crops,objective):
    """Actual grad-enabled training arithmetic, before any optimizer update."""
    if len(crops) != ACCUMULATION or len({c['source_id'] for c in crops}) != ACCUMULATION:
        raise ValueError('The pre-update probe requires12 distinct fitting sources')
    named = model.group_named_parameters(); params = [p for _,p in named]
    state = state_hash(model.decoder)
    with replay.diagnostic_state_guard(model,teacher):
        denom = base.reconstruction_denominators(crops,objective)
        model.zero_grad(set_to_none=True)
        totals = dict.fromkeys(prior.COEFFICIENTS,0.)
        rows = []
        for crop in crops:
            z,t,valid,spans = base.batch([crop])
            with torch.no_grad():
                trace = base.teacher_forward(teacher,z)
                hn = model.group_from_input(trace['group_input'])
                pn = model.suffix_from_group(hn)
                native = native_decoder_forward(teacher.model.decoder,z)
            with torch.enable_grad():
                h = model.group_from_input(trace['group_input'])
                p = model.suffix_from_group(h)
                branches = base.losses(p,t,h,trace['group_output'],valid,spans,objective,denom)
                total = sum(prior.COEFFICIENTS[key]*value for key,value in branches.items())
                if not torch.isfinite(total): raise RuntimeError('Nonfinite pre-update objective')
                total.backward()
            rows.append({'source_id':crop['source_id'],
                'grad_vs_nograd_group':compare_tensors(h,hn),
                'grad_vs_nograd_wave':compare_tensors(p,pn),
                'grad_vs_native':compare_tensors(p,native,valid),
                'grad_vs_cache':cache_check(crop,p),
                'quiet_grad_vs_cache':zero_safe_metrics(p,t,valid),
                'losses':{key:float(value.detach()) for key,value in branches.items()}})
            for key,value in branches.items(): totals[key] += float(value.detach())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError('Missing or nonfinite pre-update gradients')
        gradients = {name:{'max_abs':float(p.grad.abs().max()),'l2':float(p.grad.double().norm())} for name,p in named}
        result = {'rows':rows,'losses':totals,'parameter_gradients':gradients,
                  'all_gradients_zero':all(row['max_abs']==0 for row in gradients.values()),
                  'all_losses_zero':all(value==0 for value in totals.values()),
                  'all_grad_vs_nograd_exact':all(row['grad_vs_nograd_wave']['exact'] and row['grad_vs_nograd_group']['exact'] for row in rows),
                  'parameter_updates':0,'student_training':model.training}
    if state_hash(model.decoder) != state: raise RuntimeError('Pre-update probe changed the model')
    return result


def stability_step(model,teacher,crops,objective,optimizer):
    if len(crops) != ACCUMULATION or len({c['source_id'] for c in crops}) != ACCUMULATION:
        raise ValueError('Control stability requires12 distinct singleton sources')
    named = model.group_named_parameters()
    before = {name:p.detach().clone() for name,p in named}
    values = screen.training_update(model,teacher,crops,'current',objective,prior.COEFFICIENTS,
                                    optimizer,record_diagnostics=True)
    parameters = {}
    for name,p in named:
        if p.grad is None or not torch.isfinite(p.grad).all() or not torch.isfinite(p).all():
            raise RuntimeError('Nonfinite or missing control gradient')
        delta = p.detach()-before[name]
        parameters[name] = {'gradient_max_abs':float(p.grad.abs().max()),
            'gradient_l2':float(p.grad.double().norm()),'update_max_abs':float(delta.abs().max()),
            'update_l2':float(delta.double().norm()),'unchanged':bool(torch.equal(p,before[name]))}
    return {'losses':values,'parameters':parameters,
            'all_gradients_zero':all(v['gradient_max_abs']==0 for v in parameters.values()),
            'all_parameters_unchanged':all(v['unchanged'] for v in parameters.values()),
            'all_losses_zero':all(values[key]==0 for key in ('total','waveform','mel','feature')),
            'source_ids':[c['source_id'] for c in crops],
            'optimizer_steps':sorted({float(v['step']) for v in optimizer.state.values()})}


def final_run_guard(checkpoint_path, source_ids):
    """Fail before model loading unless the60000-source5000 run completed."""
    from progressive_continue_5000 import VERSION as TRAIN_VERSION
    path = Path(checkpoint_path)
    receipt = json.loads(path.with_suffix('.json').read_text())
    completed = json.loads((path.parent/'completed.json').read_text())
    launch = json.loads((path.parent/'launch.json').read_text())
    checksum = base.sha(path)
    if (completed.get('version') != TRAIN_VERSION or completed.get('status') != 'awaiting_review'
            or completed.get('failure') is not None or completed.get('step') != 5000
            or completed.get('cut_updates') != 5000 or completed.get('sources_seen') != TOTAL_SOURCES
            or completed.get('last_checkpoint_sha256') != checksum
            or completed.get('frozen_state_preserved') is not True or completed.get('original_files_preserved') is not True
            or receipt.get('checkpoint_sha256') != checksum or receipt.get('cut_updates') != 5000
            or receipt.get('global_updates') != 5000 or receipt.get('sources_seen') != TOTAL_SOURCES
            or receipt.get('frozen_state_preserved') is not True or launch.get('target_step') != 5000):
        raise ValueError('The original5000-step training run has not completed with intact receipts')
    payload = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (payload.get('format') != TRAIN_VERSION or payload.get('identity') != launch
            or payload.get('cut_index') != 1 or payload.get('cut_updates') != 5000
            or payload.get('global_updates') != 5000 or payload.get('accumulation') != 12
            or payload.get('coefficients') != prior.COEFFICIENTS
            or payload.get('sources_seen') != list(source_ids) or len(source_ids) != TOTAL_SOURCES
            or len(set(source_ids)) != TOTAL_SOURCES
            or payload.get('teacher_source_sha256') != base.SOURCE_SHA256
            or payload.get('teacher_checkpoint_sha256') != base.CHECKPOINT_SHA256):
        raise ValueError('Completed checkpoint source, recipe or teacher lineage differs')
    prior.validate_recipe(launch)
    for filename,expected in launch['protected'].items():
        if base.sha(filename) != expected: raise ValueError('Preserved input changed: '+filename)
    return {'checkpoint_sha256':checksum,'source_ids_sha256':screen.digest(list(source_ids)),
            'source_count':len(source_ids),'step':5000,'launch':launch,
            'protected':{str(p.resolve()):base.sha(p) for p in
                (path,path.with_suffix('.json'),path.parent/'completed.json',path.parent/'launch.json')}}


def region_levels(windows):
    """Unchanged seven overlapping cohorts, with absolute signal and DC levels."""
    from joint_recovery_gates_v2 import summarize_regions
    result = summarize_regions(windows)
    selects = {
        'all_quiet':lambda w:True,
        'near_silence':lambda w:w['teacher_rms']<=1e-5,
        'near_startup_first20ms':lambda w:w['teacher_rms']<=1e-5 and w['source_start_sample']<960,
        'source_zero_20to40ms':lambda w:w['source_reference_exact_zero'] and 960<=w['source_start_sample']<1920,
        'source_zero_after40ms':lambda w:w['source_reference_exact_zero'] and w['source_start_sample']>=1920,
        'near_after800ms':lambda w:w['teacher_rms']<=1e-5 and w['source_start_sample']>=38400,
        'quiet_nonzero_reference':lambda w:not w['source_reference_exact_zero'],
    }
    for name,select in selects.items():
        rows = [row for row in windows if select(row)]
        count = sum(row['valid_samples'] for row in rows)
        if count != result['regions'][name]['samples']: raise RuntimeError('Quiet cohort selection changed')
        for key in ('prediction_mean','teacher_mean','residual_mean'):
            result['regions'][name][key] = sum(row[key]*row['valid_samples'] for row in rows)/count if count else None
        result['regions'][name]['window_dc_residual_rms'] = math.sqrt(
            sum(row['residual_mean']**2*row['valid_samples'] for row in rows)/count) if count else None
    return result


def _write_rows(path, rows):
    with gzip.open(path,'wt') as handle:
        for row in rows: handle.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')


def cached_self_baseline(crops,out):
    rows = []
    for crop in crops:
        _,full,_,spans = base.batch([crop])
        a,b = spans[0]; target = full[...,a:b]
        raw = quiet_metrics(target,target,torch.ones_like(target,dtype=torch.bool))
        rows.extend(joint.quiet_audit.describe_quiet_windows(target,target,crop,raw))
    result = {'reference':'cached target against itself; no neural forward',
              'sources':len(crops),'quiet_regions':region_levels(rows),
              'all_quiet_passed':all(row['passed'] for row in rows),'parameter_updates':0,
              'device':str(target.device),'dtype':str(target.dtype),'grid':'same base.batch singleton scored span as production evaluator'}
    _write_rows(out/'cached-self-windows.jsonl.gz',rows)
    base.write_json(out/'cached-self-baseline.json',result)
    return result


class NativeTeacherView:
    """Feed native outputs into the existing evaluator without mutating teacher."""
    def __init__(self): self.group = self.wave = None
    def group_from_input(self,x):
        if self.group is None: raise RuntimeError('Native teacher output was not captured')
        return self.group
    def suffix_from_group(self,h):
        if h is not self.group: raise RuntimeError('Native group result was replaced')
        return self.wave


@torch.no_grad()
def evaluate_panel(model,teacher,crops,objective,metadata,out,label,*,native=False):
    from unified_monitor import UnifiedMonitor
    windows = []
    def summarize(rows):
        windows.extend(rows)
        return region_levels(rows)
    original = base.teacher_forward
    view = NativeTeacherView() if native else model
    if native:
        def observed(t,z):
            trace = original(t,z)
            with capture_boundaries(t.model.decoder) as values:
                view.wave = native_decoder_forward(t.model.decoder,z)
            view.group = values['stage4_output']
            return trace
        base.teacher_forward = observed
    try:
        result = joint.evaluate_review(UnifiedMonitor(None,{},metadata),view,teacher,crops,objective,summarize)
    finally:
        base.teacher_forward = original
    result['prediction_role'] = 'original native teacher' if native else 'independent original-weight student'
    result['target_role'] = 'immutable full-source cached teacher target'
    _write_rows(out/(label+'-windows.jsonl.gz'),windows)
    base.write_json(out/(label+'.json'),result)
    return result


def warm_shapes(teacher,model,crops,warmed,evidence=None):
    """Three initial forwards per shape and grad mode; no backward or updates."""
    count = 0
    for crop in crops:
        shape = tuple(crop['latents'].shape)
        if shape in warmed: continue
        with replay.diagnostic_state_guard(model,teacher):
            z,_,_,_ = base.batch([crop])
            key = (tuple(z.shape),str(z.dtype),str(z.device))
            internal_warm = key not in getattr(teacher,'_compression_warmed_shapes',set())
            with torch.no_grad():
                for iteration in range(3):
                    native_decoder_forward(teacher.model.decoder,z)
                    normal = model.forward_from_latents(z)
                    if iteration == 0:
                        cold_normal = {k:normal[k].detach().clone() for k in ('group_output','waveform')}
                last_normal = {k:normal[k].detach().clone() for k in ('group_output','waveform')}
                tr = base.teacher_forward(teacher,z)
            model.train(True)
            with torch.enable_grad():
                for iteration in range(3):
                    h = model.group_from_input(tr['group_input']); p = model.suffix_from_group(h)
                    if iteration == 0:
                        cold_grad = {'group_output':h.detach().clone(),'waveform':p.detach().clone()}
                last_grad = {'group_output':h.detach(),'waveform':p.detach()}
            comparisons = {}
            for label,a,b in (('first_vs_third_nograd',cold_normal,last_normal),
                              ('first_grad_vs_first_nograd',cold_grad,cold_normal),
                              ('first_grad_vs_third_nograd',cold_grad,last_normal),
                              ('first_vs_third_grad',cold_grad,last_grad)):
                for name in ('group_output','waveform'):
                    comparisons[label+'/'+name] = compare_tensors(a[name],b[name])
            if evidence is not None:
                evidence.append({'shape':list(shape),'source_id':crop['source_id'],
                    'all_exact':all(row['exact'] for row in comparisons.values()),
                    'comparisons':comparisons,'parameter_updates':0,
                    'no_extra_forwards_for_comparison':True})
            # Six native/full-wrapper warmups, three grad-enabled group/suffix
            # warmups, plus the teacher trace and any internal trace warmups.
            count += 10 + (3 if internal_warm else 0)
        warmed.add(shape)
    return count


def gpu_idle_snapshot():
    command = ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']
    result = subprocess.run(command,capture_output=True,text=True,check=True)
    if result.stdout.strip(): raise RuntimeError('GPU has another compute process; run only after training exits: '+result.stdout.strip())
    process = subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True)
    active = [line for line in process.stdout.splitlines()
              if 'python' in line and any(token in line for token in
                 ('progressive_train.py','progressive_continue.py','progressive_continue_5000.py','produce_progressive_extension_5000.py'))]
    if active: raise RuntimeError('Training/producer process still active: '+'; '.join(active))
    return {'gpu_compute_processes':[],'matching_training_or_producer_processes':[],
            'query':'nvidia-smi compute processes and ps command lines'}


def compact_comparison(value):
    return {key:value[key] for key in ('elements','exact','allclose_existing','max_abs',
                                     'residual_rms','mean_residual','expected_rms','actual_rms')}


def compact_cache(value):
    return {key:compact_comparison(value[key]) for key in ('valid','context','right_padding')}


def cache_quiet_summary(crop,prediction):
    a=crop['context_frames']*1920; b=a+crop['valid_scored_samples']
    p=prediction[...,a:b]; t=crop['teacher_audio'][...,a:b].to(p.device)
    raw=quiet_metrics(p,t,torch.ones_like(p,dtype=torch.bool))
    rows=[w for w in raw['windows'] if w['is_quiet']]
    n=sum(w['valid_samples'] for w in rows)
    error=sum(w['valid_samples']*w['residual_rms']**2 for w in rows)
    near=[w for w in rows if w['teacher_rms']<=1e-5]
    return {'quiet_windows':len(rows),'quiet_failed':sum(w['passed'] is False for w in rows),
            'quiet_samples':n,'quiet_squared_error':error,'quiet_residual_rms':math.sqrt(error/n) if n else None,
            'near_windows':len(near),'near_failed':sum(w['passed'] is False for w in near)}


def main():
    from fresh_training_data import FreshTrainingData
    from progressive_extended_data import ExtendedProgressiveData
    parser = argparse.ArgumentParser()
    for name in ('checkpoint','manifest','source-plan','shards','extension-plan','extension-shards','assets','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Identity control requires a new isolated output directory')
    if args.out.resolve().is_relative_to(args.checkpoint.parent.resolve()):
        raise ValueError('Control output cannot overlap the completed training run')
    # This read-only completion check occurs before policy() can initialize CUDA.
    completion = json.loads((args.checkpoint.parent/'completed.json').read_text())
    if completion.get('status') != 'awaiting_review' or completion.get('step') != 5000:
        raise ValueError('Wait until the existing5000-step continuation completes')
    process = gpu_idle_snapshot()
    manifest,pools,cache_receipt = base.load_data(args.manifest)
    original = prior.SourceStream(pools,FreshTrainingData(args.source_plan,args.manifest,pools,args.shards))
    data = ExtendedProgressiveData(original,args.extension_plan,args.extension_shards)
    authenticated = final_run_guard(args.checkpoint,data.source_ids)
    if len(pools['development']) != 96 or len(pools['calibration']) != 72:
        raise ValueError('The original fixed evaluation pools changed')
    paths = [args.manifest,args.source_plan,args.extension_plan,args.assets/'audio_vae_v2.py',
             args.assets/'audiovae.pth',Path(__file__),Path(group.__file__),Path(progressive.__file__),
             Path(prior.__file__),Path(base.__file__),Path(screen.__file__),Path(joint.__file__)]
    protected = {**authenticated['launch']['protected'],**authenticated['protected'],
                 **{str(path.resolve()):base.sha(path) for path in paths}}
    args.out.mkdir(parents=True)
    base.write_json(args.out/'launch.json',{'version':VERSION,'completion_guard':authenticated,
        'process_snapshot':process,'source_count':TOTAL_SOURCES,'stability_updates_max':STABILITY_UPDATES,
        'accumulation':ACCUMULATION,'coefficient_policy':prior.COEFFICIENTS,
        'reference_A':'original native decoder, same latent/context and48k sample-rate condition',
        'reference_B':'existing immutable full-source cached targets',
        'parameter_updates_in_cache_audit':0,'automatic_promotion':False,'protected':protected,
        'storage_policy':'Compact60000-source gzip JSONL; full stage detail only96-panel and first20 mismatches; no model or waveform archives',
        'expected_compressed_audit_bytes_upper_estimate':100_000_000})
    started = time.monotonic(); status = 'running'; failure = None
    model = teacher = None; teacher_hash = control_hash = None
    count = updates = warmups = 0; no_update_bad = 0; cache_nonexact = cache_outside = 0
    panel_cache_nonexact = panel_cache_outside = 0
    state_report = None; panel_before = panel_after = probe = None
    quiet_failed = near_failed = failure_details = 0
    warmup_evidence = []
    try:
        base.policy()
        if (str(torch.__version__) != authenticated['launch']['torch']
                or torch.backends.cudnn.version() != authenticated['launch']['cudnn']):
            raise ValueError('Use the preserved5000 runtime and backend')
        teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
        independently_loaded = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
        model = build_control(independently_loaded.model.decoder)
        del independently_loaded
        if tuple(len(model.selections[key]) for key in progressive.KEYS) != (512,256):
            raise RuntimeError('The control must have original512/256 widths')
        state_report = state_identity(teacher.model.decoder,model)
        base.write_json(args.out/'initial-state.json',state_report)
        if not state_report['equal'] or not state_report['independent_storage']:
            raise RuntimeError('Independent original-state or storage identity failed')
        teacher_hash, control_hash = state_hash(teacher.model), state_hash(model.decoder)
        teacher_versions = resume.continuation.teacher_versions(teacher)
        frozen = screen.frozen_versions(model)
        optimizer = progressive.fresh_optimizer(model)
        common = base.objective()
        metadata = {row['source_id']:row for row in manifest['splits']['development']['rows']}
        development = pools['development']; warmed = set()
        cached_self = cached_self_baseline(development,args.out)
        if not cached_self['all_quiet_passed']: raise RuntimeError('Cached target against itself failed quiet metrics')
        saved_quality = json.loads((args.checkpoint.parent/'development-step5000.json').read_text())
        if cached_self['quiet_regions']['window_identity_sha256'] != saved_quality['quiet_regions']['window_identity_sha256']:
            raise RuntimeError('GPU self-baseline teacher/window identity differs from preserved5000 scoring')
        warmups += warm_shapes(teacher,model,development,warmed,warmup_evidence)
        _write_rows(args.out/'warmup-parity.jsonl.gz',warmup_evidence)
        # Both native and wrapper pathways are independently checked with
        # native output hooks; a common trace helper is not the sole oracle.
        with gzip.open(args.out/'development-forward-identity.jsonl.gz','wt') as raw:
            for crop in development:
                z,_,_,_ = base.batch([crop])
                mode_results = {}
                for mode in (False,True):
                    model.train(mode)
                    result,native,wave = compare_forward(teacher.model.decoder,model,z,return_tensors=True)
                    result['native_cache'] = cache_check(crop,native)
                    result['student_cache'] = cache_check(crop,wave)
                    mode_results[str(mode)] = result
                    no_update_bad += int(not result['all_exact'])
                    if not mode:
                        panel_cache_nonexact += int(not result['native_cache']['valid']['exact'])
                        panel_cache_outside += int(not result['native_cache']['valid']['allclose_existing'])
                raw.write(json.dumps({'source_id':crop['source_id'],'modes':mode_results},allow_nan=False)+'\n')
        model.eval()
        native_report = evaluate_panel(model,teacher,development,common,metadata,args.out,'native-teacher-before',native=True)
        panel_before = evaluate_panel(model,teacher,development,common,metadata,args.out,'control-before')
        for value in (native_report,panel_before):
            if value['quiet_regions']['window_identity_sha256'] != cached_self['quiet_regions']['window_identity_sha256']:
                raise RuntimeError('Native/student evaluation changed the fixed quiet teacher/windows')
        if state_hash(model.decoder) != control_hash or state_hash(teacher.model) != teacher_hash:
            raise RuntimeError('No-update development evaluation altered original state')
        fitting = data.take(0,STABILITY_UPDATES*ACCUMULATION)
        if len({c['source_id'] for c in fitting}) != len(fitting): raise RuntimeError('Stability source repetition')
        warmups += warm_shapes(teacher,model,fitting[:ACCUMULATION],warmed,warmup_evidence)
        _write_rows(args.out/'warmup-parity.jsonl.gz',warmup_evidence)
        model.train(True)
        probe = training_forward_probe(model,teacher,fitting[:ACCUMULATION],common)
        base.write_json(args.out/'grad-enabled-pre-update.json',probe)
        model.eval()
        if no_update_bad:
            status = 'no_update_mismatch'
            raise RuntimeError('Fixed-panel native/wrapper or trace identity failed before the large audit')
        # One source-ordered pass. The reader retains only its current shard;
        # compressed JSONL retains every comparison without waveform/model copies.
        with gzip.open(args.out/'consumed-source-audit.jsonl.gz','wt') as raw, gzip.open(args.out/'first-mismatch-details.jsonl.gz','wt') as details:
            for start in range(0,TOTAL_SOURCES,300):
                crops = data.take(start,min(300,TOTAL_SOURCES-start))
                if [c['source_id'] for c in crops] != list(data.source_ids[start:start+len(crops)]):
                    raise RuntimeError('Consumed source cache order changed')
                earlier_shapes = len(warmup_evidence)
                warmups += warm_shapes(teacher,model,crops,warmed,warmup_evidence)
                if len(warmup_evidence) != earlier_shapes:
                    _write_rows(args.out/'warmup-parity.jsonl.gz',warmup_evidence)
                for crop in crops:
                    z,_,_,_ = base.batch([crop])
                    result,native,wave = compare_forward(teacher.model.decoder,model,z,include_trace=False,return_tensors=True,detailed_boundaries=False)
                    native_cache, student_cache = cache_check(crop,native),cache_check(crop,wave)
                    no_update_bad += int(not result['all_exact'])
                    cache_nonexact += int(not native_cache['valid']['exact'])
                    cache_outside += int(not native_cache['valid']['allclose_existing'])
                    quiet=cache_quiet_summary(crop,native); quiet_failed+=quiet['quiet_failed']; near_failed+=quiet['near_failed']
                    compact={'source_index':count,'source_id':crop['source_id'],'cache_key':crop.get('cache_key'),
                        'cache_binding':'original manifest and pooled-cache SHA' if count<3000 else 'sealed source-plan/shard receipt',
                        'start_frame':crop['start_frame'],'context_frames':crop['context_frames'],
                        'valid_scored_samples':crop['valid_scored_samples'],
                        'native_vs_student':compact_comparison(result['comparisons']['native_vs_student']),
                        'native_cache':compact_cache(native_cache),'student_cache':compact_cache(student_cache),'quiet':quiet}
                    raw.write(json.dumps(compact,allow_nan=False,separators=(',',':'))+'\n'); count += 1
                    if failure_details<20 and (not result['all_exact'] or not native_cache['valid']['exact']):
                        full=compare_forward(teacher.model.decoder,model,z)
                        full.update(source_index=count-1,source_id=crop['source_id'],native_cache=native_cache,student_cache=student_cache)
                        details.write(json.dumps(full,allow_nan=False)+'\n');failure_details+=1
                raw.flush()
                progress = {'sources_audited':count,'total_sources':TOTAL_SOURCES,'native_wrapper_nonexact':no_update_bad,
                    'valid_cache_nonexact':cache_nonexact,'valid_cache_outside_existing_tolerance':cache_outside,
                    'quiet_failed_windows':quiet_failed,'near_failed_windows':near_failed,'phase':'no_update_cache_audit',
                    'compressed_audit_bytes':(args.out/'consumed-source-audit.jsonl.gz').stat().st_size,
                    'elapsed_seconds':time.monotonic()-started,'parameter_updates':0}
                base.write_json(args.out/'audit-progress.json',progress);base.event('identical_cache_audit',**progress)
        data.assert_unchanged()
        if no_update_bad or cache_outside or panel_cache_outside:
            status = 'no_update_mismatch'; raise RuntimeError('No-update identity/cache mismatch; exact flags retained and optimizer probe skipped')
        if state_hash(model.decoder) != control_hash or state_hash(teacher.model) != teacher_hash:
            raise RuntimeError('No-update complete-cache scan altered model state')
        model.train(True)
        if cache_nonexact or panel_cache_nonexact: status = 'cache_sensitive'
        # The actual recipe is retained even if the grad-enabled backend differs.
        # The first drift ends this bounded diagnostic, and postmetrics are saved.
        with (args.out/'stability-updates.jsonl').open('w') as raw:
            for index in range(STABILITY_UPDATES):
                crops = fitting[index*ACCUMULATION:(index+1)*ACCUMULATION]
                evidence = stability_step(model,teacher,crops,common,optimizer)
                updates += 1; evidence['update'] = updates
                raw.write(json.dumps(evidence,allow_nan=False)+'\n');raw.flush()
                if not (evidence['all_losses_zero'] and evidence['all_gradients_zero'] and evidence['all_parameters_unchanged']):
                    status = 'training_path_drift'; break
        model.eval()
        panel_after = evaluate_panel(model,teacher,development,common,metadata,args.out,'control-after')
        if status == 'running':
            sensitive = (not all(row['all_exact'] for row in warmup_evidence)
                         or not probe['all_grad_vs_nograd_exact'] or not probe['all_gradients_zero'] or not probe['all_losses_zero'])
            status = 'grad_mode_sensitive' if sensitive else 'passed'
        if screen.frozen_versions(model) != frozen or resume.continuation.teacher_versions(teacher) != teacher_versions:
            raise RuntimeError('Diagnostic changed a frozen model component')
    except BaseException as exc:
        failure = repr(exc)
        if status == 'running': status = 'failed'
        raise
    finally:
        unchanged = all(base.sha(path)==value for path,value in protected.items())
        data.assert_unchanged()
        teacher_preserved = teacher_hash is None or state_hash(teacher.model)==teacher_hash
        control_final_hash = state_hash(model.decoder) if model is not None else None
        result = {'version':VERSION,'status':status,'failure':failure,'sources_audited':count,
            'native_wrapper_nonexact':no_update_bad,'valid_cache_nonexact':cache_nonexact,
            'valid_cache_outside_existing_tolerance':cache_outside,'stability_updates':updates,
            'development_valid_cache_nonexact':panel_cache_nonexact,
            'development_valid_cache_outside_existing_tolerance':panel_cache_outside,
            'quiet_failed_windows_during_audit':quiet_failed,'near_failed_windows_during_audit':near_failed,
            'extra_warmup_forwards':warmups,'teacher_preserved':teacher_preserved,
            'warmup_shapes':len(warmup_evidence),'warmup_nonexact_shapes':sum(not r['all_exact'] for r in warmup_evidence),
            'cold_grad_nonexact_shapes':sum(any(not v['exact'] for k,v in r['comparisons'].items()
                if k.startswith(('first_grad_','first_vs_third_grad/'))) for r in warmup_evidence),
            'original_files_preserved':unchanged,'initial_control_sha256':control_hash,
            'final_control_sha256':control_final_hash,'control_state_unchanged':control_final_hash==control_hash,
            'grad_mode_precheck':{k:probe[k] for k in ('all_gradients_zero','all_losses_zero','all_grad_vs_nograd_exact')} if probe else None,
            'before':panel_before['aggregate'] if panel_before else None,
            'after':panel_after['aggregate'] if panel_after else None,
            'elapsed_seconds':time.monotonic()-started,'automatic_promotion':False}
        base.write_json(args.out/'completed.json',result)
        if not unchanged or not teacher_preserved: raise RuntimeError('Identity diagnostic changed protected originals')


if __name__ == '__main__': main()
