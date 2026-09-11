"""Bounded startup/Q diagnosis. Aggregate-only output; original files untouched."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import quiet_candidate_recovery as recovery
import quiet_projected_update as quiet
import progressive_train as prior
import fresh_training_data

base, screen, replay = prior.base, prior.screen, prior.replay


def build(config):
    row = next(s for s in config['stages'] if s['name'] == 'quiet_recovery')
    argv = row['argv'][2:]
    values = {}
    for k, v in zip(argv[::2], argv[1::2]):
        key = k[2:].replace('-', '_')
        values[key] = v if key.endswith('sha256') else Path(v)
    args = SimpleNamespace(**values)
    base.policy()
    manifest, pools, _ = base.load_data(args.manifest)
    fresh = fresh_training_data.FreshTrainingData(args.source_plan, args.manifest, pools, args.shards)
    data = prior.SourceStream(pools, fresh)
    loaded = recovery.authenticate_inputs(args, data, pools)
    zero, ba, br, ca, cr, _, recipe, _, _, _, protected = loaded
    if str(torch.__version__) != recipe['torch'] or torch.backends.cudnn.version() != recipe['cudnn']:
        raise RuntimeError('Qualified runtime differs')
    recovery.control.gpu_idle_snapshot()
    teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py', args.assets/'audiovae.pth', device='cuda')
    model = recovery.progressive.initialize_from_teacher(teacher.model.decoder, ca['selection'])
    optimizer, _ = recovery.restore_quiet_start(model, teacher, zero, ba, br, ca, cr,
        [c['source_id'] for c in pools['calibration']], artifact_sha256=args.combined_artifact_sha256)
    if recovery.control.state_hash(model.decoder) != args.combined_state_sha256:
        raise RuntimeError('Fresh C state differs')
    return args, teacher, model, optimizer, pools, data, protected


def prepare(teacher, crops):
    entries = []
    with replay.observe_teacher_cache(crops) as checks:
        for crop in crops:
            z, target, _, _ = base.batch([crop])
            entry = quiet.window_layout(crop, target)
            trace = base.teacher_forward(teacher, z)
            entry['group_input'] = trace['group_input'].detach()
            entries.append(entry)
    if len(checks) != len(crops) or not all(c['allclose_original_tolerance'] for c in checks):
        raise RuntimeError('Teacher cache validation failed')
    return entries, {'comparisons':len(checks), 'all_pass':True,
        'nonexact':sum(not c['bitwise_equal'] for c in checks)}


def difference(x, y):
    d = x.detach().double()-y.detach().double()
    return {'exact':bool(torch.equal(x,y)), 'max_abs':float(d.abs().max()),
            'rms':float(d.square().mean().sqrt()), 'elements':d.numel()}


def forward(model, entry, enabled, trace=False):
    captures = {}; handles = []
    if trace:
        for name, module in model.decoder.named_modules():
            if name and not list(module.children()):
                def hook(m, inputs, output, name=name):
                    if isinstance(output, torch.Tensor):captures[name]=output.detach().clone()
                handles.append(module.register_forward_hook(hook))
    try:
        with torch.set_grad_enabled(enabled):
            prediction = quiet._predict(model, entry)
            records = quiet.window_excesses(prediction, entry)
        return prediction, records, captures
    finally:
        for h in handles:h.remove()


def values(records):
    return [float(v.detach()) for r in records for v in r['values']]


def numeric(model, teacher, crops, optimizer):
    entries, cache = prepare(teacher, crops)
    params = base.parameters(model)
    state = recovery.control.state_hash(model.decoder)
    rng = screen.rng_state()
    warmed=set();prior.warm_student(model,teacher,crops,warmed,optimizer)
    summary=[]; first=None
    for i,e in enumerate(entries):
        if not e['windows']:continue
        p0,r0,cold0=forward(model,e,False,True)
        p1,r1,cold1=forward(model,e,True,True)
        a,b=values(r0),values(r1)
        row={'quiet_windows':len(r0),'prediction':difference(p0,p1),
             'constraint_mismatches':sum(x!=y for x,y in zip(a,b)),
             'constraint_max_abs_difference':max((abs(x-y) for x,y in zip(a,b)),default=0.),
             'acceptance_changes':sum(x['passed']!=y['passed'] for x,y in zip(r0,r1))}
        summary.append(row)
        if first is None and row['constraint_mismatches']:
            first=i
            cold_layers=[{'module':k,**difference(cold0[k],cold1[k])} for k in cold0 if k in cold1]
        del p0,p1,r0,r1,cold0,cold1
    detail=None
    if first is not None:
        e=entries[first]
        with torch.no_grad():p0,r0,t0=forward(model,e,False,True)
        p1,r1,t1=forward(model,e,True,True)
        pp,rp,_=forward(model,e,True)
        pn,rn,_=forward(model,e,False)
        layers=[{'module':k,**difference(t0[k],t1[k])} for k in t0 if k in t1]
        # Re-score identical detached predictions through both metric paths.
        detached=p0.detach()
        with torch.no_grad():same0=quiet.window_excesses(detached,e)
        with torch.enable_grad():same1=quiet.window_excesses(detached.clone().requires_grad_(True),e)
        detail={'cold_first_different_module':next((r['module'] for r in cold_layers if not r['exact']),None),
                'cold_layer_differences':cold_layers,
                'first_different_module':next((r['module'] for r in layers if not r['exact']),None),
                'layer_differences':layers,'repeat_grad_prediction':difference(p1,pp),
                'repeat_no_grad_prediction':difference(p0,pn),
                'repeat_grad_constraint_exact':values(r1)==values(rp),
                'repeat_no_grad_constraint_exact':values(r0)==values(rn),
                'same_prediction_metric_exact':values(same0)==values(same1)}
        del p0,p1,pp,pn,r0,r1,rp,rn,t0,t1,same0,same1
    screen.restore_rng(rng)
    if recovery.control.state_hash(model.decoder)!=state or any(p.grad is not None for p in params):
        raise RuntimeError('Numerical diagnostic altered state or gradients')
    return {'teacher_cache':cache,'crops':len(crops),'quiet_sources':len(summary),
            'quiet_windows':sum(r['quiet_windows'] for r in summary),
            'prediction_nonexact_sources':sum(not r['prediction']['exact'] for r in summary),
            'prediction_max_abs_difference':max((r['prediction']['max_abs'] for r in summary),default=0.),
            'constraint_mismatches':sum(r['constraint_mismatches'] for r in summary),
            'constraint_max_abs_difference':max((r['constraint_max_abs_difference'] for r in summary),default=0.),
            'acceptance_changes':sum(r['acceptance_changes'] for r in summary),
            'first_mismatch_detail':detail,'state_preserved':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('New diagnostic output required')
    started=time.monotonic()
    _,teacher,model,optimizer,pools,data,protected=build(json.loads(args.config.read_text()))
    teacher_hash=recovery.control.state_hash(teacher.model)
    result=numeric(model,teacher,data.take(0,12),optimizer)
    result.update(version='startup_retention_numerical_audit_v2',elapsed_seconds=time.monotonic()-started,
                  fresh_adam_empty=not optimizer.state,teacher_preserved=recovery.control.state_hash(teacher.model)==teacher_hash,
                  protected_files_unchanged=all(base.sha(p)==sha for p,sha in protected.items()),
                  scope='First original fitting batch only; no optimizer updates; no raw audio, latents, source identities or weights exported')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'state_preserved':result['state_preserved'],
                      'quiet_sources':result['quiet_sources'],'output':str(args.out)}))


if __name__=='__main__':main()
