"""Read-only four-site accounting after the same-coefficient precision control.

Teacher-coordinate restorations are restricted to authenticated sliced step0.
The recovered model is observed only; this program has no optimizer or fit.
All hidden-boundary comparisons remain reported under their original tolerance.
The required restoration contract is local accounting plus full stage4 and wave.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import gzip
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import diagnose_channel_contributions_v2 as common
import identical_teacher_control as control
import progressive_model as progressive

group, base, prior = common.group, common.base, control.prior
VERSION = 'audiovae2_progressive_silence_v2'
STEP0_SHA = 'bd9a09c1ab5e86fce8f5ba1f435565c9dfff76b83c32d413d037a54d0df5942f'
STEP5000_SHA = '4dd64e0d165ab23aa56e5c9f0e0fe0d281e510dfb3db8d6cd863a2be0640fe47'
ATOL, RTOL = 1e-5, 1e-4


def restoration_acceptance(boundary_checks, local_accounting):
    """Keep strict hidden flags separate from the required external contract.

    This gate follows the preserved-coefficient FP64 control, which showed
    arithmetic amplification inside the frozen suffix. No threshold changes.
    """
    names = ('stage2_ru1','stage2_ru2','stage2_ru3','stage3_up')
    if set(local_accounting) != set(names): raise ValueError('Expected all four local accounting sites')
    local_ok = True
    for name in names:
        entry = local_accounting[name]
        keys = ('linear_identity',) if name=='stage3_up' else ('linear_identity','residual_identity')
        for key in keys:
            checks = entry[key]
            if set(checks) != {'teacher_sum','gap_sum'}: raise ValueError('Incomplete local accounting identity')
            local_ok = local_ok and all(checks[k]['passed'] is True for k in checks)
    required = ('stage4_output','waveform')
    if any(k not in boundary_checks for k in required): raise ValueError('Missing required full group or waveform comparison')
    hidden = {k:v for k,v in boundary_checks.items() if k not in required}
    failed = [k for k,v in hidden.items() if v['allclose_existing'] is not True]
    return {'required_contract_passed':local_ok and all(boundary_checks[k]['allclose_existing'] is True for k in required),
            'local_accounting_passed':local_ok,'required_boundaries':list(required),
            'hidden_boundaries_allclose':not failed,'failed_hidden_boundaries':failed,
            'all_boundary_checks_passed':not failed and all(boundary_checks[k]['allclose_existing'] is True for k in required),
            'scope':'Original local identities and full stage4/wave tolerance are required; hidden pointwise failures remain findings, not equality claims'}


def evaluate_with_quiet_capture(monitor,model,teacher,crops,objective,path,evaluate_fn=None):
    """One unchanged evaluation; observe every quiet row without extra forwards."""
    from joint_recovery_gates_v2 import summarize_regions
    calls = 0
    def capture(rows):
        nonlocal calls
        calls += 1
        if calls != 1: raise RuntimeError('Expected exactly one quiet summary per evaluation')
        control._write_rows(Path(path),rows)
        return summarize_regions(rows)
    fn = control.joint.evaluate_review if evaluate_fn is None else evaluate_fn
    result = fn(monitor,model,teacher,crops,objective,summarize=capture)
    if calls != 1: raise RuntimeError('Evaluation did not provide its original quiet rows')
    return result


def verify_precision_proof(path):
    """Require the bounded same-coefficient proof before this amended gate."""
    path = Path(path)
    completed = json.loads(path.read_text())
    results = json.loads((path.parent/'precision-results.json').read_text())
    coefficients = json.loads((path.parent/'coefficient-preservation.json').read_text())
    expected = {'as_in:train:13453488421825608782.wav',
                'thorsten_emotional:whisper/37773d61d5c91c12d49e55e2b041c0cd.wav'}
    if (completed.get('version') != 'audiovae2_progressive_restoration_precision_v1'
            or completed.get('complete') is not True or completed.get('failure') is not None
            or completed.get('sources') != 2 or completed.get('all_boundaries_original_tolerance') is not True
            or completed.get('files_preserved') is not True or not completed.get('states_preserved')
            or not all(completed['states_preserved'].values())
            or coefficients.get('all_fp32_coefficients_preserved_exactly') is not True
            or coefficients.get('student_resliced_in_fp64') is not False
            or len(results) != 2 or {r['source_id'] for r in results} != expected):
        raise ValueError('The preserved-coefficient FP64 proof has not completed successfully')
    for result in results:
        if (result.get('all_boundaries_original_tolerance') is not True
                or len(result['restored_local_mixers']) != 4
                or not all(v['allclose_existing'] is True for v in result['restored_local_mixers'].values())):
            raise ValueError('FP64 local or boundary accounting did not pass the original tolerance')
    return {'status':'original_tolerance_closure','tight_fp64_passed':completed['all_boundaries_tight_fp64'],
            'actual_fp32_coefficients_preserved':True,'original_step0':coefficients['original_step0'],
            'files':{str(p.resolve()):base.sha(p) for p in
                (path,path.parent/'precision-results.json',path.parent/'coefficient-preservation.json',path.parent/'launch.json')}}


def four_sites(teacher_decoder, selection):
    width2 = teacher_decoder.model[3].block[1].out_channels
    width3 = teacher_decoder.model[4].block[1].out_channels
    k2 = group._indices(selection['stage2_indices'], width2, 'stage2').tolist()
    k3 = group._indices(selection['stage3_indices'], width3, 'stage3').tolist()
    if len(k2) >= width2 or k3 != list(range(width3)):
        raise ValueError('This diagnostic requires only stage2 narrowed and unchanged ordered stage3')
    sites = [common.Operation(f'stage2_ru{i}', f'model.3.block.{i+1}.block.3', k2, k2, 1200)
             for i in range(1, 4)]
    up = teacher_decoder.model[4].block[1]
    if up.stride != (5,) or up.kernel_size != (10,):
        raise ValueError('Stage3 upsampler must preserve stride5/kernel10')
    sites.append(common.Operation('stage3_up', 'model.4.block.1', k2, k3, 6000, 5))
    return sites


def _versions(module):
    return tuple((name, id(t), t.data_ptr(), t._version) for name,t in
                 list(module.named_parameters()) + list(module.named_buffers()))


def validate_pristine(teacher_decoder, student, selection):
    """Match all raw tensors to a fresh teacher-derived slice before any hook.

    The private version/storage token invalidates authorization after mutation.
    Final whole-state hashing separately authenticates preservation.
    """
    four_sites(teacher_decoder, selection)
    expected = progressive.initialize_from_teacher(teacher_decoder, selection).to(
        device=next(student.parameters()).device, dtype=next(student.parameters()).dtype)
    a, b = expected.decoder.state_dict(), student.decoder.state_dict()
    if list(a) != list(b) or any(a[k].shape != b[k].shape or a[k].dtype != b[k].dtype
                               or not torch.equal(a[k], b[k]) for k in a):
        raise ValueError('Restoration requires pristine original-teacher sliced initialization')
    del expected
    return {'passed': True, 'student_state_sha256': control.state_hash(student.decoder),
            'teacher_state_sha256': control.state_hash(teacher_decoder),
            '_student_id': id(student), '_student_versions': _versions(student.decoder),
            '_teacher_id': id(teacher_decoder), '_teacher_versions': _versions(teacher_decoder)}


def _check_pristine(student, teacher_decoder, report):
    if (not report.get('passed') or report.get('_student_id') != id(student)
            or report.get('_teacher_id') != id(teacher_decoder)
            or report.get('_student_versions') != _versions(student.decoder)
            or report.get('_teacher_versions') != _versions(teacher_decoder)):
        raise ValueError('Pristine restoration authorization is stale or belongs to another model')


@contextmanager
def restoration(student, teacher_decoder, teacher_capture, sites, names, *, pristine_report):
    _check_pristine(student, teacher_decoder, pristine_report)
    if len(set(names)) != len(names) or not set(names).issubset(s.name for s in sites):
        raise ValueError('Unknown or duplicated restoration site')
    with ExitStack() as stack:
        for site in sites:
            if site.name not in names: continue
            missing = common.dropped_response(teacher_decoder.get_submodule(site.path),
                                               teacher_capture[site.name]['input'], site).detach()
            def add(output, value=missing):
                if output.shape != value.shape: raise ValueError('Restoration output geometry differs')
                return output + value
            stack.enter_context(common.temporary_output_change(student.decoder.get_submodule(site.path), add))
        try:
            yield
        finally:
            _check_pristine(student, teacher_decoder, pristine_report)


@contextmanager
def capture_sites(decoder, sites):
    """Capture linear branch inputs/outputs plus each residual skip exactly once."""
    with common.capture_operations(decoder, sites) as captured:
        handles = []
        try:
            for site in sites:
                if site.stride != 1: continue
                def hook(module, args, output, name=site.name):
                    captured[name]['skip_input'] = args[0].detach()
                    captured[name]['ru_output'] = output.detach()
                parent = decoder.get_submodule(site.path.rsplit('.block.3', 1)[0])
                handles.append(parent.register_forward_hook(hook))
            yield captured
        finally:
            for handle in handles: handle.remove()


def decompose_site(site, teacher_decoder, student, teacher_capture, student_capture):
    t, s = teacher_capture[site.name], student_capture[site.name]
    terms = common.operation_decomposition(teacher_decoder.get_submodule(site.path),
        student.decoder.get_submodule(site.path), t['input'], s['input'], t['output'], s['output'],
        site.kept_inputs, site.kept_outputs)
    return terms


def residual_terms(site, terms, teacher_capture, student_capture):
    """Complete RU accounting adds skip drift to the pre-skip linear gap."""
    if site.stride != 1: raise ValueError('Residual accounting requires a pointwise RU site')
    t, s = teacher_capture[site.name], student_capture[site.name]
    skip = t['skip_input'][:, site.kept_outputs]
    skip_drift = skip - s['skip_input']
    return {'kept': terms['kept'] + skip, 'dropped': terms['dropped'], 'bias': terms['bias'],
            'teacher_selected': t['ru_output'][:, site.kept_outputs], 'student': s['ru_output'],
            'retained_drift': terms['retained_drift'], 'skip_drift': skip_drift,
            'gap': t['ru_output'][:, site.kept_outputs] - s['ru_output'],
            'reconstructed_gap': terms['reconstructed_gap'] + skip_drift}


def interval_mask(valid, start, stop):
    if valid.dtype != torch.bool or valid.ndim != 3 or valid.shape[:2] != (1, 1):
        raise ValueError('Expected singleton waveform validity mask')
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= valid.shape[-1]:
        raise ValueError('Invalid waveform interval')
    out = torch.zeros_like(valid); out[..., start:stop] = valid[..., start:stop]
    return out


def absolute_interval(crop, start, stop):
    """Translate scored-wave offsets to full tensor and original-source samples."""
    n, context = crop['valid_scored_samples'], crop['context_frames']
    if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= n
            or crop['start_frame'] - crop['context_start_frame'] != context):
        raise ValueError('Invalid scored interval or latent context geometry')
    return {'tensor_start': context*1920 + start, 'tensor_stop': context*1920 + stop,
            'source_start': crop['start_frame']*1920 + start,
            'source_stop': crop['start_frame']*1920 + stop}


def window_summary(prediction, target, start, stop):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[:2] != (1, 1):
        raise ValueError('Expected equal singleton waveform tensors')
    mask = interval_mask(torch.ones_like(target, dtype=torch.bool), start, stop)
    result = control.compare_tensors(prediction, target, mask)
    p, t = prediction[..., start:stop].detach().double(), target[..., start:stop].detach().double()
    d = p-t; dc = float(d.mean()); mse = float(d.square().mean())
    pp, tt, dot = float(p.square().sum()), float(t.square().sum()), float((p*t).sum())
    quiet = common.diagnostic_quiet_metrics(prediction[...,start:stop],target[...,start:stop],
        torch.ones_like(target[...,start:stop],dtype=torch.bool))
    return {**result, 'teacher_mean': float(t.mean()), 'student_mean': float(p.mean()),
            'dc_error': dc, 'dc_mean_square': dc*dc,
            'ac_mean_square': float((d-dc).square().mean()), 'residual_mean_square': mse,
            'dc_energy_fraction': dc*dc/mse if mse else None,
            'mae': float(d.abs().mean()), 'rms_ratio': math.sqrt(pp/tt) if tt else None,
            'least_squares_gain': dot/tt if tt else None,
            'cosine': dot/math.sqrt(pp*tt) if pp*tt else None,
            'maximum_error_offset_samples':int(d.abs().reshape(-1).argmax()),
            'quiet_metric':quiet}


def resolve_panel_rows(panel, crops, observed_windows):
    """Resolve sealed source-absolute windows without looking at predictions.

    observed_windows includes quiet and active rows on the production scored grid.
    Exact full13 startup coverage is enforced by the production caller.
    """
    if panel.get('version') != 'progressive_silence_panel_v1':
        raise ValueError('Unknown fixed-window panel version')
    crop_map = {c['source_id']: c for c in crops}
    observed = {(r['source_id'], r['source_start_sample'], r['source_stop_sample']): r
                for r in observed_windows}
    rows, seen = [], set()
    for item in panel['rows']:
        key = (item['source_id'], item['source_start_sample'], item['source_stop_sample'])
        if key in seen or key not in observed or key[0] not in crop_map:
            raise ValueError('Duplicated, absent or shifted fixed panel window')
        seen.add(key); crop = crop_map[key[0]]
        a, b = key[1]-crop['start_frame']*1920, key[2]-crop['start_frame']*1920
        coordinates = absolute_interval(crop, a, b)
        row = dict(item)
        row.update(coordinates, scored_start=a, scored_stop=b,
                   teacher_rms=observed[key]['teacher_rms'], is_quiet=observed[key]['is_quiet'])
        if not isinstance(row.get('roles'), list) or not row['roles']:
            raise ValueError('Every fixed window needs a predeclared selection role')
        rows.append(row)
    if not rows or len(rows) > 256 or len({r['source_id'] for r in rows}) > 24:
        raise ValueError('Fixed panel exceeds the bounded protocol')
    return rows


def load_panel(panel, crops):
    quiet_rows, observed = [], []
    for crop in crops:
        _, target, _, spans = base.batch([crop]); a,b = spans[0]; t = target[..., a:b]
        raw = common.diagnostic_quiet_metrics(t,t,torch.ones_like(t,dtype=torch.bool))
        quiet_rows.extend(control.joint.quiet_audit.describe_quiet_windows(t,t,crop,raw))
        for row in raw['windows']:
            observed.append({**row, 'source_id':crop['source_id'],
                'source_start_sample':crop['start_frame']*1920+row['start_sample'],
                'source_stop_sample':crop['start_frame']*1920+row['stop_sample']})
    identity = control.region_levels(quiet_rows)['window_identity_sha256']
    if panel['window_identity_sha256'] != identity:
        raise ValueError('Sealed panel differs from the fixed GPU teacher-window identity')
    rows = resolve_panel_rows(panel,crops,observed)
    crop_map = {c['source_id']:c for c in crops}
    for row in rows:
        roles = row['roles']
        if 'same_source_earliest_complete_active_after40ms' in roles:
            active = [w for w in observed if w['source_id']==row['source_id'] and not w['is_quiet']
                      and w['source_start_sample']>=1920 and w['source_stop_sample']-w['source_start_sample']==960]
            if not active or row['source_start_sample'] != min(w['source_start_sample'] for w in active):
                raise ValueError('Sealed same-source active control differs from the teacher-only rule')
        elif not row['is_quiet']:
            raise ValueError('A preselected quiet window is active under the unchanged GPU metric')
        if any('near' in role for role in roles) and row['teacher_rms'] > 1e-5:
            raise ValueError('Preselected near-silence window is not teacher-near')
        if 'startup_near_all13' in roles and (row['source_start_sample'],row['source_stop_sample']) != (0,960):
            raise ValueError('Startup role is not the absolute first20ms window')
        if any('source_zero' in role for role in roles):
            reference = control.joint.quiet_audit.source_reference_window(crop_map[row['source_id']],row['scored_start'],row['scored_stop'])
            if not reference['source_reference_exact_zero']:
                raise ValueError('Preselected source-zero control contains nonzero original16k samples')
    startup = {(r['source_id'],r['source_start_sample'],r['source_stop_sample']) for r in observed
               if r['teacher_rms'] <= 1e-5 and r['source_start_sample'] < 960}
    actual = {(r['source_id'],r['source_start_sample'],r['source_stop_sample']) for r in rows}
    if len(startup) != 13 or not startup.issubset(actual):
        raise ValueError('The diagnostic must cover all13 fixed near-silence startup windows')
    return rows, {'window_identity_sha256':identity,'startup_windows':13,
                  'selected_windows':len(rows),'selected_sources':len({r['source_id'] for r in rows})}


def boundary_summary(prediction, target, mask):
    if prediction.shape != target.shape:
        raise ValueError('Shared-boundary comparison requires all original output channels')
    weights = common.cell_weights(mask, prediction.shape[-1])
    p,t = prediction.detach().double(),target.detach().double()
    d = p-t; summary = common.weighted_tensor_summary(d,weights)
    return {'error':summary, 'teacher':common.weighted_tensor_summary(t,weights),
            'student':common.weighted_tensor_summary(p,weights),
            'scope':'Full shared output channels; channel-wise temporal means define DC within this fixed window'}


def latent_summary(z, rows):
    value = z.detach().double()
    result = {'frames':value.shape[-1],'channels':value.shape[1],
              'first_frame_rms':float(value[...,0].square().mean().sqrt()),
              'first_frame_mean':float(value[...,0].mean()),'all_exact_zero':bool((value==0).all()),
              'temporal_difference_rms':float(value.diff(dim=-1).square().mean().sqrt()) if value.shape[-1]>1 else None}
    selected = []
    for row in rows:
        first,stop = row['tensor_start']//1920,(row['tensor_stop']+1919)//1920
        x = value[...,first:stop]
        selected.append({'source_start_sample':row['source_start_sample'],'latent_frame_span':[first,stop],
                         'rms':float(x.square().mean().sqrt()),'mean':float(x.mean()),
                         'exact_zero':bool((x==0).all())})
    return {**result,'selected_windows':selected}


@torch.no_grad()
def warm_models(teacher, models, crops):
    seen = set(); count = 0
    for crop in crops:
        shape = tuple(crop['latents'].shape)
        if shape in seen: continue
        z,_,_,_ = base.batch([crop]); tr = base.teacher_forward(teacher,z)
        for model in models:
            for _ in range(3): model.suffix_from_group(model.group_from_input(tr['group_input']))
            count += 3
        seen.add(shape)
    return {'extra_student_warmup_forwards':count,'latent_shapes':len(seen),
            'teacher_warmup':'Existing teacher_forward warms3 times per new singleton shape'}


def verify_control_completion(path):
    receipt = json.loads(Path(path).read_text())
    if (receipt.get('status') not in ('passed','cache_sensitive','grad_mode_sensitive','training_path_drift')
            or receipt.get('failure') is not None or receipt.get('sources_audited') != 60000
            or receipt.get('native_wrapper_nonexact') != 0
            or receipt.get('valid_cache_outside_existing_tolerance') != 0
            or receipt.get('development_valid_cache_outside_existing_tolerance') != 0
            or receipt.get('teacher_preserved') is not True
            or receipt.get('original_files_preserved') is not True):
        raise ValueError('Complete the original-teacher control without identity/outside-tolerance failure first')
    return receipt


def checkpoint_inputs(args):
    if base.sha(args.step0) != STEP0_SHA or base.sha(args.checkpoint) != STEP5000_SHA:
        raise ValueError('Use the exact preserved current-cut step0 and step5000 checkpoints')
    zero = torch.load(args.step0,map_location='cpu',weights_only=True,mmap=True)
    trained = torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    authenticated = control.final_run_guard(args.checkpoint,trained['sources_seen'])
    if (zero.get('cut_updates') != 0 or zero.get('cut_index') != 1
            or zero.get('selection') != trained.get('selection')
            or zero.get('schedule_sha256') != trained.get('schedule_sha256')
            or tuple(len(trained['selection'][k]) for k in progressive.KEYS) != (384,256)):
        raise ValueError('Original sliced initialization and current384/256 model lineage differ')
    if base.sha(args.assets/'audio_vae_v2.py') != base.SOURCE_SHA256 or base.sha(args.assets/'audiovae.pth') != base.CHECKPOINT_SHA256:
        raise ValueError('Original teacher asset hashes differ')
    return zero,trained,authenticated


def _public_pristine(report):
    return {k:v for k,v in report.items() if not k.startswith('_')}


@torch.no_grad()
def main():
    from compare_accumulation_v1 import numeric_reference_check
    from unified_monitor import UnifiedMonitor
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('step0','checkpoint','manifest','assets','panel','control-completed','precision-proof','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Use a new isolated diagnostic output directory')
    for protected_dir in (args.step0.parent,args.checkpoint.parent,args.control_completed.parent):
        if args.out.resolve().is_relative_to(protected_dir.resolve()):
            raise ValueError('Diagnostic output must not overlap preserved runs')
    control_receipt = verify_control_completion(args.control_completed)
    precision_proof = verify_precision_proof(args.precision_proof)
    zero,trained,authenticated = checkpoint_inputs(args)
    process = control.gpu_idle_snapshot()
    manifest,pools,cache_receipt = base.load_data(args.manifest)
    if len(pools['development']) != 96: raise ValueError('Expected the unchanged96-source development panel')
    panel = json.loads(args.panel.read_text())
    paths = [args.step0,args.checkpoint,args.manifest,args.panel,args.control_completed,
             args.step0.parent/'development-step0.json',args.checkpoint.parent/'development-step5000.json',
             args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',
             Path(__file__),Path(common.__file__),Path(control.__file__),Path(group.__file__),
             Path(progressive.__file__),Path(prior.__file__),Path(base.__file__),
             Path(manifest['cache_path']),Path(manifest['overlay_receipt_path'])]
    protected = {**authenticated['protected'],**precision_proof['files'],**{str(p.resolve()):base.sha(p) for p in paths}}
    base.policy()
    if (str(torch.__version__) != authenticated['launch']['torch']
            or torch.backends.cudnn.version() != authenticated['launch']['cudnn']):
        raise ValueError('Use the preserved5000 runtime and backend')
    selected_rows,panel_receipt = load_panel(panel,pools['development'])
    args.out.mkdir(parents=True)
    base.write_json(args.out/'launch.json',{'version':VERSION,'protected':protected,
        'step0_sha256':STEP0_SHA,'step5000_sha256':STEP5000_SHA,'selection':trained['selection'],
        'control_status':control_receipt['status'],'control_cache_nonexact':control_receipt['valid_cache_nonexact'],
        'control_warmup_nonexact_shapes':control_receipt.get('warmup_nonexact_shapes'),
        'precision_proof':precision_proof,
        'restoration_acceptance':'Unchanged local identities plus full stage4 and waveform tolerance; all hidden-boundary strict flags retained separately',
        'panel':panel_receipt,'process_snapshot':process,'backend':common.replay.backend_state(),
        'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),
        'parameter_updates':0,'fit_or_optimization':False,'automatic_promotion':False,
        'restoration_scope':'Only authenticated original teacher-derived sliced step0; never adapted5000',
        'storage':'Compressed per-source JSON and selected waveform snippets; no weights or full recordings saved'})
    base.write_json(args.out/'panel.json',{'identity':panel_receipt,'rows':selected_rows})
    started = time.monotonic(); failure = None; teacher = initial = recovered = full = None
    hashes = {}; finished = 0; checks = []; baseline_checks = {}; warmup = None
    traces = {}; trace_index = []
    try:
        teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
        initial = progressive.initialize_from_teacher(teacher.model.decoder,zero['selection'])
        initial.load_group_state_dict(zero['group']); initial.eval()
        recovered = progressive.initialize_from_teacher(teacher.model.decoder,trained['selection'])
        recovered.load_group_state_dict(trained['group']); recovered.eval()
        pristine = validate_pristine(teacher.model.decoder,initial,zero['selection'])
        for key in ('teacher_state_sha256','student_state_sha256'):
            if precision_proof['original_step0'][key] != pristine[key]:
                raise ValueError('FP64 proof teacher or saved step0 state differs from this diagnostic')
        full = control.build_control(teacher.model.decoder).eval()
        copy_identity = control.state_identity(teacher.model.decoder,full)
        if not copy_identity['equal'] or not copy_identity['independent_storage']:
            raise RuntimeError('Full-width independent-state control failed')
        base.write_json(args.out/'model-identities.json',{'pristine':_public_pristine(pristine),'full_copy':copy_identity})
        objects = {'teacher':teacher.model,'step0':initial.decoder,'step5000':recovered.decoder,'full_copy':full.decoder}
        hashes = {name:control.state_hash(model) for name,model in objects.items()}
        development = pools['development']; objective = base.objective()
        metadata = {row['source_id']:row for row in manifest['splits']['development']['rows']}
        warmup = warm_models(teacher,[initial,recovered],development)
        for step,model,path in ((0,initial,args.step0.parent),(5000,recovered,args.checkpoint.parent)):
            report = evaluate_with_quiet_capture(UnifiedMonitor(None,{},metadata),model,teacher,development,objective,
                args.out/f'quiet-windows-step{step}.jsonl.gz')
            saved = json.loads((path/f'development-step{step}.json').read_text())
            baseline_checks[str(step)] = numeric_reference_check(report,saved)
            base.write_json(args.out/f'development-step{step}.json',report)
            base.write_json(args.out/'saved-baseline-parity.json',baseline_checks)
            if not baseline_checks[str(step)]['passed']:
                raise RuntimeError(f'Saved step{step} baseline metric reproduction failed')
        sites = four_sites(teacher.model.decoder,zero['selection'])
        by_source = {}
        for row in selected_rows: by_source.setdefault(row['source_id'],[]).append(row)
        chosen = [crop for crop in development if crop['source_id'] in by_source]
        full_warm = warm_models(teacher,[full],chosen)
        warmup['full_copy_extra_student_warmup_forwards'] = full_warm['extra_student_warmup_forwards']
        with gzip.open(args.out/'source-diagnostics.jsonl.gz','wt') as raw:
            for crop in chosen:
                rows = by_source[crop['source_id']]; z,target,valid,_ = base.batch([crop])
                with capture_sites(teacher.model.decoder,sites) as tc, control.capture_boundaries(teacher.model.decoder) as tb:
                    native = control.native_decoder_forward(teacher.model.decoder,z)
                tb['waveform'] = native
                teacher_trace = group.teacher_trace(teacher.model.decoder,z)
                native_check = control.compare_tensors(teacher_trace['waveform'],native)
                with control.capture_boundaries(full.decoder) as fb:
                    full_wave = full.forward_from_latents(z)['waveform']
                fb['waveform'] = full_wave
                copy_checks = {name:control.compare_tensors(fb[name],value) for name,value in tb.items()}
                if not native_check['exact'] or not all(value['exact'] for value in copy_checks.values()):
                    raise RuntimeError('Native/trace/full-width control lost exact numerical identity')
                with capture_sites(initial.decoder,sites) as sc, control.capture_boundaries(initial.decoder) as ib:
                    plain = initial.forward_from_latents(z)['waveform']
                ib['waveform'] = plain
                with control.capture_boundaries(recovered.decoder) as rb:
                    trained_wave = recovered.forward_from_latents(z)['waveform']
                rb['waveform'] = trained_wave
                cache = control.cache_check(crop,native)
                if not cache['valid']['allclose_existing']:
                    raise RuntimeError('Native teacher and cached target exceed the unchanged tolerance')
                observations = {'source_id':crop['source_id'],'context_start_frame':crop['context_start_frame'],
                    'start_frame':crop['start_frame'],'context_frames':crop['context_frames'],
                    'valid_scored_samples':crop['valid_scored_samples'],'native_trace':native_check,
                    'full_copy_comparisons':copy_checks,'native_cache':cache,'latents':latent_summary(z,rows),
                    'operations':{},'windows':[]}
                masks = {str(i):interval_mask(valid,row['tensor_start'],row['tensor_stop']) for i,row in enumerate(rows)}
                masks.update(common.region_masks(target,valid,crop))
                for site in sites:
                    terms = decompose_site(site,teacher.model.decoder,initial,tc,sc)
                    entry = {'linear_pre_skip':common.summarize_terms(terms,masks,site.stride),
                             'linear_identity':common.assert_decomposition(terms,site.name),
                             'scope':'Original sliced step0 only; retained-input drift includes propagated prior omissions'}
                    if site.stride == 1:
                        ru = residual_terms(site,terms,tc,sc)
                        entry['complete_residual_unit'] = common.summarize_terms(ru,masks)
                        entry['residual_identity'] = common.assert_decomposition(ru,site.name+'/with_skip')
                    observations['operations'][site.name] = entry
                variants = {'teacher':native,'step0':plain,'step5000':trained_wave}
                restore_boundaries = None
                for name,names in [(s.name,[s.name]) for s in sites]+[('all_four',[s.name for s in sites])]:
                    with restoration(initial,teacher.model.decoder,tc,sites,names,pristine_report=pristine):
                        with control.capture_boundaries(initial.decoder) as current:
                            wave = initial.forward_from_latents(z)['waveform']
                    variants['step0_restore_'+name] = wave
                    if name == 'all_four': restore_boundaries = {**current,'waveform':wave}
                restore_checks = {}
                for name,tensor in tb.items():
                    expected = tensor
                    if name == 'stage2_output': expected = tensor[:,zero['selection']['stage2_indices']]
                    elif name == 'stage3_conditioned_input': expected = tensor[:,zero['selection']['stage2_indices']]
                    restore_checks[name] = control.compare_tensors(restore_boundaries[name],expected)
                observations['all_four_restoration_identity'] = restore_checks
                for i,row in enumerate(rows):
                    a,b = row['tensor_start'],row['tensor_stop']; mask = masks[str(i)]
                    window = {**row,'reference16k':control.joint.quiet_audit.source_reference_window(crop,row['scored_start'],row['scored_stop']),
                        'waveforms':{name:window_summary(wave,target,a,b) for name,wave in variants.items()},
                        'waveforms_vs_native':{name:window_summary(wave,native,a,b) for name,wave in variants.items() if name!='teacher'},
                        'shared_boundaries':{}}
                    for name in ('stage1_output','stage3_output','stage4_output','stage5_output','stage6_output','pre_tanh','waveform'):
                        window['shared_boundaries'][name] = {
                            'step0':boundary_summary(ib[name],tb[name],mask),
                            'step5000':boundary_summary(rb[name],tb[name],mask)}
                    key = f'source{finished}_window{i}'
                    for name,wave in {**variants,'cached_target':target}.items():
                        traces[key+'_'+name] = wave[0,0,a:b].detach().cpu().numpy().copy()
                    trace_index.append({'key':key,'source_id':crop['source_id'],
                        'source_samples':[row['source_start_sample'],row['source_stop_sample']],
                        'tensor_samples':[a,b],'sample_rate':48000,'roles':row['roles']})
                    observations['windows'].append(window)
                raw.write(json.dumps(observations,allow_nan=False,separators=(',',':'))+'\n');raw.flush()
                finished += 1
                check = {'source_id':crop['source_id'],**restoration_acceptance(restore_checks,observations['operations']),
                         'all_four_waveform':restore_checks['waveform'],'full_copy_exact':True}
                checks.append(check);base.write_json(args.out/'restoration-checks.json',checks)
                base.write_json(args.out/'progress.json',{'sources':finished,'expected_sources':len(chosen),
                    'elapsed_seconds':time.monotonic()-started,'parameter_updates':0})
                if not check['required_contract_passed']:
                    raise RuntimeError('Required local/full-group/wave restoration contract failed; raw evidence preserved')
        np.savez_compressed(args.out/'selected-waveforms.npz',**traces)
        base.write_json(args.out/'trace-index.json',trace_index)
    except BaseException as exc:
        failure = repr(exc)
        if traces:
            np.savez_compressed(args.out/'selected-waveforms.npz',**traces)
            base.write_json(args.out/'trace-index.json',trace_index)
        raise
    finally:
        files_preserved = all(base.sha(path)==checksum for path,checksum in protected.items())
        state_preserved = {name:control.state_hash(model)==hashes[name] for name,model in objects.items()} if hashes else {}
        passed = failure is None and files_preserved and bool(state_preserved) and all(state_preserved.values())
        base.write_json(args.out/'completed.json',{'version':VERSION,'status':'completed' if passed else 'failed',
            'failure':failure,'sources_diagnosed':finished,'panel':panel_receipt,'baselines':baseline_checks,
            'required_restoration_contracts_passed':bool(checks) and all(row['required_contract_passed'] for row in checks),
            'hidden_boundaries_allclose':bool(checks) and all(row['hidden_boundaries_allclose'] for row in checks),
            'hidden_boundary_failure_sources':{r['source_id']:r['failed_hidden_boundaries'] for r in checks if r['failed_hidden_boundaries']},
            'original_files_preserved':files_preserved,'state_preserved':state_preserved,'initial_state_hashes':hashes,
            'warmup':warmup,'elapsed_seconds':time.monotonic()-started,'parameter_updates':0,'automatic_promotion':False})
        if not files_preserved or (hashes and not all(state_preserved.values())):
            raise RuntimeError('Read-only diagnostic altered a protected file or model tensor')


if __name__ == '__main__': main()
