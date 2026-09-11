"""Instrumented exact replay of the already-completed4500-to5000 segment.

Original checkpoints, recipe and source order are immutable. This is diagnosis,
not continuation beyond the existing5000-step model or a new training setting.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import time

import torch
import resume_settings as resume
from fresh_training_data import FreshTrainingData

screen, base = resume.screen, resume.base
VERSION = 'audiovae2_exact_late_replay_v1'
CASE_IDS = (
    'es_419:train:13461728374156750135.wav',
    'kn_in:train:15096009457771558023.wav',
    'kashmiri:1970324837177077_chunk_1.flac',
    'freesound:428921',
)


def compare_tree(actual, expected, path='root'):
    """Compare values/dtypes/shapes and nested scalar types, never archive bytes."""
    result = {'equal': True, 'tensors': 0, 'scalars': 0, 'mismatches': []}
    def bad(where, reason, **extra):
        result['equal'] = False
        if len(result['mismatches']) < 30:
            result['mismatches'].append({'path': where, 'reason': reason, **extra})
    def visit(a, b, where):
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            result['tensors'] += 1
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): bad(where, 'tensor/type'); return
            if a.dtype != b.dtype or a.shape != b.shape: bad(where, 'tensor shape/dtype'); return
            aa, bb = a.detach().cpu(), b.detach().cpu()
            if not torch.equal(aa, bb):
                extra = {}
                if aa.numel() and (aa.is_floating_point() or aa.dtype != torch.bool):
                    delta = (aa.double()-bb.double()).abs()
                    extra['max_abs'] = float(delta.max()) if bool(torch.isfinite(delta).all()) else None
                bad(where, 'tensor values', **extra)
        elif type(a) is not type(b): bad(where, 'scalar/container type')
        elif isinstance(a, dict):
            if set(a) != set(b): bad(where, 'dictionary keys'); return
            for key in a: visit(a[key], b[key], where+'.'+str(key))
        elif isinstance(a, (tuple, list)):
            if len(a) != len(b): bad(where, 'sequence length'); return
            for i, (x, y) in enumerate(zip(a, b)): visit(x, y, where+f'[{i}]')
        else:
            result['scalars'] += 1
            if isinstance(a, float) and (not math.isfinite(a) or not math.isfinite(b)): bad(where, 'nonfinite scalar')
            elif a != b: bad(where, 'scalar value', actual=a, expected=b)
    visit(actual, expected, path)
    return result


def replay_window(start_payload, end_payload, original_ids, fresh_ids):
    if start_payload.get('step') != 4500 or end_payload.get('step') != 5000:
        raise ValueError('Only the completed4500-to5000 segment is allowed')
    start = resume.validate_source_ledger(start_payload, original_ids, fresh_ids, 3)
    stop = resume.validate_source_ledger(end_payload, original_ids, fresh_ids, 3)
    if (start, stop) != (10500, 12000): raise ValueError('Wrong replay source interval')
    for key in ('identity', 'resume_identity'):
        if start_payload.get(key) != end_payload.get(key): raise ValueError('Replay endpoint recipe differs')
    configuration = start_payload['resume_identity']
    for key, expected_value in (('effective_batch_size', 3), ('execution_batch_size', 1), ('accumulation_steps', 3)):
        if configuration.get(key) != expected_value:
            raise ValueError('Replay requires original singleton accumulation contract')
    return start, stop


def validate_training_record(step, chosen_ids, actual_values, original_row):
    expected = {key: original_row[key] for key in ('total', 'waveform', 'mel', 'feature')}
    if 'gradient_norm' in original_row: expected['gradient_norm'] = original_row['gradient_norm']
    observed = {key: actual_values[key] for key in expected}
    return compare_tree({'step': step, 'source_ids': list(chosen_ids), 'values': observed},
                        {'step': original_row['step'], 'source_ids': original_row['source_ids'], 'values': expected},
                        path=f'step{step}')


def backend_state():
    return {'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}


def restore_backend(s):
    torch.backends.cudnn.enabled = s['cudnn_enabled']
    torch.backends.cudnn.benchmark = s['cudnn_benchmark']
    torch.backends.cudnn.deterministic = s['cudnn_deterministic']
    torch.backends.cudnn.allow_tf32 = s['cudnn_tf32']
    torch.backends.cuda.matmul.allow_tf32 = s['matmul_tf32']
    torch.use_deterministic_algorithms(s['deterministic'], warn_only=s['deterministic_warn_only'])


@contextmanager
def diagnostic_state_guard(model, teacher, optimizer=None):
    """Restore diagnostic RNG/modes/grads and reject parameter/state mutation."""
    modules = [model] + ([] if teacher is None else [teacher])
    snapshots = [type(module.state_dict())((k, v.detach().clone()) for k, v in module.state_dict().items()) for module in modules]
    parameters = [(p, p.requires_grad, p.grad, None if p.grad is None else p.grad.detach().clone())
                  for module in modules for p in module.parameters()]
    modes = [(m, m.training) for module in modules for m in module.modules()]
    rng, backend = screen.rng_state(), backend_state()
    opt = None if optimizer is None else copy.deepcopy(optimizer.state_dict())
    changed = []
    try:
        yield
    finally:
        with torch.no_grad():
            for i, (module, snapshot) in enumerate(zip(modules, snapshots)):
                if not compare_tree(module.state_dict(), snapshot)['equal']:
                    module.load_state_dict(snapshot); changed.append(f'module{i}')
            if optimizer is not None and not compare_tree(optimizer.state_dict(), opt)['equal']:
                optimizer.load_state_dict(opt); changed.append('optimizer')
            for p, requires, original_grad, saved_grad in parameters:
                p.requires_grad_(requires)
                p.grad = original_grad
                if original_grad is not None and not torch.equal(original_grad, saved_grad): original_grad.copy_(saved_grad)
            for module, training in modes: module.training = training
        restore_backend(backend); screen.restore_rng(rng)
        if changed: raise RuntimeError('Diagnostic mutated protected state: '+','.join(changed))


def replay_step(model, teacher, crops, common, coefficients, optimizer, record_diagnostics, update_fn=None):
    params = base.parameters(model)
    before = [p.detach().clone() for p in params]
    fn = screen.training_update if update_fn is None else update_fn
    values = fn(model, teacher, crops, 'current', common, coefficients, optimizer,
                record_diagnostics=record_diagnostics)
    with torch.no_grad():
        delta_sq = torch.zeros((), device=params[0].device, dtype=torch.float64)
        gradient_sq, dot = delta_sq.clone(), delta_sq.clone()
        by_stage = {}
        named = model.group_named_parameters()
        for (name, p), old in zip(named, before):
            if p.grad is None: raise RuntimeError('Missing original post-update gradient')
            delta, gradient = p.detach().double()-old.double(), p.grad.detach().double()
            energy = delta.square().sum(); ge = gradient.square().sum(); gd = (gradient*delta).sum()
            delta_sq += energy; gradient_sq += ge; dot += gd
            parts = name.split('.'); stage = '.'.join(parts[:2])
            totals = by_stage.setdefault(stage, [torch.zeros_like(dot) for _ in range(3)])
            for i, value in enumerate((energy, ge, gd)): totals[i] += value
        ds, gs, dg = float(delta_sq), float(gradient_sq), float(dot)
        direction = {'parameter_update_l2': math.sqrt(ds), 'gradient_l2': math.sqrt(gs),
            'gradient_dot_actual_update': dg,
            'gradient_update_cosine': dg/math.sqrt(ds*gs) if ds and gs else None,
            'by_parameter_stage': {k: {'update_l2': math.sqrt(float(v[0])), 'gradient_l2': math.sqrt(float(v[1])),
                                      'gradient_dot_actual_update': float(v[2])} for k, v in by_stage.items()},
            'interpretation': 'Original weighted minibatch gradient dotted with actual AdamW parameter displacement; not a held-out gradient.'}
    return values, direction


def cached_levels(crops, metadata):
    rows = []
    for crop in crops:
        a, n = crop['context_frames']*1920, crop['valid_scored_samples']
        y = crop['teacher_audio'][0,0,a:a+n].double()
        full, tail = divmod(n, 960)
        rms = y[:full*960].reshape(full,960).square().mean(-1).sqrt()
        lengths = torch.full((full,), 960, dtype=torch.int64)
        if tail:
            rms = torch.cat((rms, y[full*960:].square().mean().sqrt().reshape(1)))
            lengths = torch.cat((lengths, torch.tensor([tail])))
        row = metadata[crop['source_id']]; source = row['manifest_row']
        rows.append({'source_id': crop['source_id'], 'language': row.get('normalized_language', source['language'].split('_')[0]),
                     'dataset': source['dataset'], 'samples': n, 'context_frames': crop['context_frames'],
                     'teacher_rms': float(y.square().mean().sqrt()), 'teacher_peak': float(y.abs().max()),
                     'quiet_samples': int(lengths[rms <= 1e-3].sum()), 'nearzero_samples': int(lengths[rms <= 1e-5].sum()),
                     'quiet_windows': int((rms <= 1e-3).sum())})
    return rows


@torch.no_grad()
def evaluate_cases(model, teacher, crops):
    rows = []
    for crop in crops:
        z, target, valid, spans = base.batch([crop])
        trace = base.teacher_forward(teacher, z)
        warmed=getattr(model,'_replay_case_warmed_shapes',set())
        key=tuple(z.shape)
        if key not in warmed:
            for _ in range(3):model.suffix_from_group(model.group_from_input(trace['group_input']))
            warmed.add(key);model._replay_case_warmed_shapes=warmed
        p = model.suffix_from_group(model.group_from_input(trace['group_input']))
        a,b = spans[0];pp,tt = p[...,a:b], target[...,a:b]
        quiet = base.quiet_window_metrics(pp, tt, torch.ones_like(pp,dtype=torch.bool))
        active = torch.zeros_like(pp, dtype=torch.bool)
        for w in quiet['windows']:
            if not w['is_quiet']: active[...,w['start_sample']:w['stop_sample']] = True
        x,y = pp.double()[active],tt.double()[active]
        te,pe = float(y.square().sum()),float(x.square().sum())
        rows.append({'source_id':crop['source_id'],'samples':b-a,'mae':float((pp.double()-tt.double()).abs().mean()),
                     'active_samples':int(active.sum()), 'active_rms_gain':math.sqrt(pe/te) if te else None,
                     'active_projection_gain':float((x*y).sum())/te if te else None,
                     'active_cosine':float((x*y).sum())/math.sqrt(pe*te) if pe and te else None,
                     'peak':float(pp.abs().max()),'quiet_failed_windows':quiet['quiet_failed_count'],
                     'teacher_cached_max_abs':float((trace['waveform'][valid]-target[valid]).abs().max())})
    return rows


@contextmanager
def observe_teacher_cache(crops):
    """Observe the original teacher forward already used by each training crop."""
    original = base.teacher_forward
    records = []
    def observe(teacher, z):
        trace = original(teacher, z)
        if len(records) >= len(crops): raise RuntimeError('Unexpected extra teacher call in training update')
        crop = crops[len(records)];a=crop['context_frames']*1920;b=a+crop['valid_scored_samples']
        target = crop['teacher_audio'][...,a:b].to(trace['waveform'].device)
        actual = trace['waveform'][...,a:b].detach()
        if actual.shape != target.shape: raise RuntimeError('Teacher/cache scored geometry differs')
        error = actual.double()-target.double()
        records.append({'source_id':crop['source_id'],'samples':b-a,'bitwise_equal':torch.equal(actual,target),
                        'allclose_original_tolerance':torch.allclose(actual,target,atol=1e-5,rtol=1e-4),
                        'max_abs':float(error.abs().max()),'residual_rms':float(error.square().mean().sqrt())})
        return trace
    base.teacher_forward = observe
    try:
        yield records
        if len(records) != len(crops): raise RuntimeError('Missing teacher/cache observations')
    finally:
        base.teacher_forward = original


def case_reference_check(rows, report):
    original={r['source_id']:r for r in report['rows']}
    checks=[]
    for row in rows:
        expected=original[row['source_id']]
        checks.append({'source_id':row['source_id'],'mae':row['mae'],'expected_mae':expected['mae'],
            'passed':row['samples']==expected['samples'] and math.isclose(row['mae'],expected['mae'],rel_tol=1e-5,abs_tol=1e-6)})
    return {'passed':all(c['passed'] for c in checks),'checks':checks,'relative_tolerance':1e-5,'absolute_tolerance':1e-6}


def main():
    parser=argparse.ArgumentParser()
    for name in ('checkpoint','expected','anchor-checkpoint','screen-out','base-out','manifest','fresh-manifest','shards','assets','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Use a new isolated replay directory')
    if args.checkpoint.resolve()==args.expected.resolve(): raise ValueError('Replay endpoints must differ')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.out.parent).free<300<<20: raise OSError('Replay needs300MiB free temporary storage')
    base.policy()
    _,selection,identity,manifest,pools,payload,_,start_sha,_=resume.authenticate(args)
    expected_sha=base.sha(args.expected)
    expected=torch.load(args.expected,map_location='cpu',weights_only=True,mmap=True)
    expected_receipt=json.loads(args.expected.with_suffix('.json').read_text())
    if (expected_sha!=expected_receipt['checkpoint_sha256'] or expected_receipt['step']!=5000
            or expected_receipt['resume_identity_sha256']!=screen.digest(expected['resume_identity'])
            or expected_receipt['frozen_state_preserved'] is not True): raise ValueError('Expected final checkpoint receipt differs')
    data=FreshTrainingData(args.fresh_manifest,args.manifest,pools,args.shards)
    original_ids=[r['source_id'] for r in pools['fit']]
    start,stop=replay_window(payload,expected,original_ids,data.source_ids)
    execution=payload['resume_identity']
    expected_execution={'precision':'float32','tf32':False,'learning_rate':3e-5,
        'approved_source_range':[0,12000], 'fresh_plan_identity_sha256':data.identity,
        'original_manifest_sha256':base.sha(args.manifest),'fresh_manifest_sha256':base.sha(args.fresh_manifest),
        'resume_sha256':base.sha(resume.__file__),
        'fresh_loader_sha256':base.sha(Path(__file__).with_name('fresh_training_data.py')),
        'monitor_sha256':base.sha(Path(__file__).with_name('unified_monitor.py'))}
    if any(execution.get(key)!=value for key,value in expected_execution.items()):
        raise ValueError('Original execution source, runtime or manifest contract changed')
    original_records=[json.loads(line) for line in (args.checkpoint.parent/'train.jsonl').read_text().splitlines()]
    record_map={r['step']:r for r in original_records}
    if len(record_map)!=len(original_records): raise ValueError('Duplicate original training journal step')
    for step in range(4501,5001):
        cursor=(step-1001)*3
        if record_map[step]['source_ids']!=list(data.source_ids[cursor:cursor+3]): raise ValueError('Original journal disagrees with sealed source order')
    development={c['source_id']:c for c in pools['development']}
    cases=[development[s] for s in CASE_IDS]
    row_metadata={r['source_id']:r for r in data.plan['rows']}
    protected_paths=(args.checkpoint,args.expected,args.anchor_checkpoint,args.manifest,args.fresh_manifest,Path(__file__),
        args.checkpoint.with_suffix('.json'),args.expected.with_suffix('.json'),args.checkpoint.parent/'train.jsonl',
        args.checkpoint.parent/'development-step4500.json',args.expected.parent/'development-step5000.json')
    protected={str(p):base.sha(p) for p in protected_paths}
    teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    model=base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices'])
    common=base.objective()
    frozen,teacher_frozen=screen.frozen_versions(model),resume.continuation.teacher_versions(teacher)
    optimizer=resume.continuation.restore_training_state(model,payload)
    coefficients=payload['identity']['coefficients']
    args.out.mkdir()
    base.write_json(args.out/'launch.json',{'version':VERSION,'start_checkpoint_sha256':start_sha,'expected_checkpoint_sha256':expected_sha,
        'source_interval':[start,stop],'updates':500,'sources':1500,'parameter_scope':'unchanged stages2-4 and their conditioning',
        'learning_rate':payload['identity']['learning_rate'],'coefficients':coefficients,'execution_batch':1,'accumulation':3,
        'case_ids':CASE_IDS,'diagnostic_every':25,'original_training_identity':payload['identity'],
        'resume_identity':payload['resume_identity'],'protected_files_sha256':protected,'backend':backend_state(),
        'no_continuation_beyond_existing_final_step':True,'exact_match_required_for_causal_attribution':True})
    started=time.monotonic();seen=list(payload['sources_seen']);comparisons=[];snapshots=[];cache_checks=[];warmed=set()
    failure=None
    try:
        with diagnostic_state_guard(model,teacher,optimizer):
            snapshots.append({'step':4500,'cases':evaluate_cases(model,teacher,cases)})
        start_case_check=case_reference_check(snapshots[0]['cases'],json.loads((args.checkpoint.parent/'development-step4500.json').read_text()))
        if not start_case_check['passed']: raise RuntimeError('Initial fixed-case MAE does not reproduce saved4500 diagnostic')
        base.write_json(args.out/'case-trajectories.json',snapshots)
        for step in range(4501,5001):
            cursor=(step-1001)*3;chosen=data.take(cursor,3)
            # Re-establish per-process forward algorithm warmup without changing
            # the saved random/gradient state or the actual optimization update.
            for crop in chosen:
                key=tuple(crop['latents'].shape)
                if key not in warmed:
                    with diagnostic_state_guard(model,teacher,optimizer),torch.no_grad():
                        z,_,_,_=base.batch([crop]);trace=base.teacher_forward(teacher,z)
                        for _ in range(3):model.suffix_from_group(model.group_from_input(trace['group_input']))
                    warmed.add(key)
            levels=cached_levels(chosen,row_metadata)
            with observe_teacher_cache(chosen) as checks:
                values,direction=replay_step(model,teacher,chosen,common,coefficients,optimizer,
                    record_diagnostics='gradient_norm' in record_map[step])
            comparison=validate_training_record(step,[c['source_id'] for c in chosen],values,record_map[step])
            comparisons.append({'step':step,**comparison});cache_checks.extend(checks)
            seen.extend(c['source_id'] for c in chosen)
            row={'step':step,'values':values,'direction':direction,'sources':levels,'original_step_equal':comparison['equal'],
                 'original_step_differences':comparison['mismatches'],'teacher_cache_checks':checks,'elapsed_seconds':time.monotonic()-started}
            with (args.out/'replay.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
            if step%25==0:
                with diagnostic_state_guard(model,teacher,optimizer):
                    snapshots.append({'step':step,'cases':evaluate_cases(model,teacher,cases)})
                base.write_json(args.out/'case-trajectories.json',snapshots)
                base.event('exact_replay_progress',step=step,all_step_values_equal=all(c['equal'] for c in comparisons),elapsed_seconds=time.monotonic()-started)
        # Original final validation also restores RNG. The fixed forward-only
        # checks above do not replace an endpoint tensor/RNG equality test.
        final_path=args.out/'replayed-step5000.pt'
        resume.save_checkpoint(final_path,model,optimizer,5000,payload['identity'],seen,payload['resume_identity'])
        replayed=torch.load(final_path,map_location='cpu',weights_only=True,mmap=True)
        final_comparison=compare_tree(replayed,expected)
        final_case_check=case_reference_check(snapshots[-1]['cases'],json.loads((args.expected.parent/'development-step5000.json').read_text()))
        for path,value in protected.items():
            if base.sha(path)!=value: raise RuntimeError('Original protected input changed: '+path)
        data.assert_unchanged()
        if screen.frozen_versions(model)!=frozen or resume.continuation.teacher_versions(teacher)!=teacher_frozen:
            raise RuntimeError('Frozen model/teacher changed during replay')
        exact=final_comparison['equal'] and all(c['equal'] for c in comparisons)
        targets_valid=all(c['allclose_original_tolerance'] for c in cache_checks)
        result={'version':VERSION,'completed_updates':500,'final_step':5000,'exact_replay':exact,
            'trajectory_valid':exact,'teacher_targets_valid':targets_valid,
            'causal_attribution_allowed':exact,'step_comparisons':comparisons,'first_step_divergence':next((c for c in comparisons if not c['equal']),None),
            'case_reference_checks':{'initial':start_case_check,'final':final_case_check},
            'final_complete_checkpoint_tree_comparison':final_comparison,'replayed_checkpoint_sha256':base.sha(final_path),
            'expected_checkpoint_sha256':expected_sha,'original_files_preserved':True,'frozen_state_preserved':True,
            'sources_exact':seen==expected['sources_seen'],'teacher_cache_sources_checked':len(cache_checks),
            'all_teacher_cache_bitwise':all(c['bitwise_equal'] for c in cache_checks),
            'all_teacher_cache_original_tolerance':all(c['allclose_original_tolerance'] for c in cache_checks),
            'teacher_cache_max_abs':max(c['max_abs'] for c in cache_checks),'teacher_cache_failures':[c for c in cache_checks if not c['allclose_original_tolerance']],
            'elapsed_seconds':time.monotonic()-started,'snapshots':snapshots,
            'limits':['Different process/GPU load may change floating-point backend choices; exact comparisons determine whether attribution is valid.',
                      'Fixed four development cases are diagnostic only and never used for optimization.',
                      'An onset coinciding with a source batch establishes temporal association, not proof that one source caused it.']}
        base.write_json(args.out/'completed.json',result)
        base.event('exact_replay_complete',exact_replay=exact,checkpoint_tree_equal=final_comparison['equal'],all_teacher_cache_bitwise=result['all_teacher_cache_bitwise'],elapsed_seconds=result['elapsed_seconds'])
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'completed_updates':len(comparisons)}
        base.write_json(args.out/'failure.json',failure)
        raise
    finally:
        base.write_json(args.out/'preservation.json',{'frozen_state_preserved':screen.frozen_versions(model)==frozen,
            'teacher_state_preserved':resume.continuation.teacher_versions(teacher)==teacher_frozen,'failure':failure})


if __name__=='__main__':main()
