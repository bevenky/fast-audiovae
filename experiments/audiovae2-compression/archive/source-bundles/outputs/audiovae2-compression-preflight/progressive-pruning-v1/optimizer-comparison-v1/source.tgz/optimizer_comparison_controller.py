"""Finite serial qualification/recovery controller. No model code is imported.

Every arm runs in a fresh subprocess. Selection consumes only the fixed training
probe; development metrics are never inspected by the selector. Existing output
directories are never resumed, replaced or removed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

VERSION = 'audiovae2_optimizer_comparison_controller_v1'
PLAN_VERSION = 'audiovae2_optimizer_comparison_plan_v1'
DECISION_VERSION = 'audiovae2_optimizer_qualification_v1'
RUNNER_VERSION = 'audiovae2_optimizer_comparison_v1'
METHODS = ('adamw', 'muon', 'normuon', 'shampoo')
GRIDS = {'adamw': (1.5e-5, 3e-5, 6e-5), 'shampoo': (1.5e-5, 3e-5, 6e-5),
         'muon': (7.5e-5, 1.5e-4, 3e-4), 'normuon': (7.5e-5, 1.5e-4, 3e-4)}
MIB = 1024**2


class TechnicalFailure(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*MIB), b''): h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path = Path(path)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')


def progress(path, value):
    # Only this controller's mutable progress receipt is replaced.
    tmp = path.with_suffix('.tmp')
    with tmp.open('x') as handle: json.dump(value, handle, indent=2, allow_nan=False)
    tmp.replace(path)


def parse_proc_stat(text):
    fields = text.rsplit(') ', 1)[1].split()
    return {'state': fields[0], 'start_ticks': fields[19]}


def proc_identity(pid):
    try: return parse_proc_stat(Path(f'/proc/{pid}/stat').read_text())
    except FileNotFoundError: return None


def resolve_source(plan, name):
    for directory in plan['pythonpath'].split(os.pathsep):
        path = Path(directory) / name
        if path.is_file(): return path.resolve()
    raise TechnicalFailure(f'Missing pinned source: {name}')


def validate_plan(plan):
    if plan.get('version') != PLAN_VERSION:
        raise ValueError('Unknown comparison plan version')
    for key in ('config', 'source_manifest', 'python', 'root', 'parity_reference'):
        if not Path(plan[key]).is_absolute(): raise ValueError(f'{key} must be absolute')
    directories = plan['pythonpath'].split(os.pathsep)
    if not directories or any(not d or not Path(d).is_absolute() for d in directories):
        raise ValueError('PYTHONPATH must contain ordered absolute directories')
    for key in ('config', 'source_manifest', 'parity_reference'):
        if sha(plan[key]) != plan[key+'_sha256']: raise TechnicalFailure(f'{key} hash changed')
    manifest = read(plan['source_manifest'])
    required = {'optimizer_comparison_runner.py', 'recovery_optimizers.py', 'startup_mixed_optimizer_update.py',
                'optimizer_comparison_controller.py'}
    if not required.issubset(manifest): raise ValueError('Incomplete executable source manifest')
    for name, expected in manifest.items():
        if Path(name).name != name or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid source manifest entry')
        if sha(resolve_source(plan, name)) != expected: raise TechnicalFailure(f'Pinned source changed: {name}')
    if sha(__file__) != manifest['optimizer_comparison_controller.py']:
        raise TechnicalFailure('Executing controller differs from sealed source')
    candidates = plan['candidates']
    if len(candidates) != 12: raise ValueError('Exactly twelve qualifications are required')
    names = set(); pairs = set()
    for candidate in candidates:
        name, method, kwargs = candidate['name'], candidate['method'], candidate['optimizer_kwargs']
        if not re.fullmatch('[a-z0-9][a-z0-9_-]{0,79}', name) or name in names:
            raise ValueError('Candidate names must be distinct safe path components')
        if method not in METHODS or set(kwargs) - {'matrix_lr', 'adam_lr'} or kwargs.get('adam_lr', 3e-5) != 3e-5:
            raise ValueError('Only the declared nine-matrix learning rate may vary')
        rate = kwargs.get('matrix_lr')
        if type(rate) not in (float, int) or rate not in GRIDS[method] or (method, rate) in pairs:
            raise ValueError('Candidate grid changed or repeated')
        names.add(name); pairs.add((method, rate))
    if pairs != {(m, r) for m in METHODS for r in GRIDS[m]}:
        raise ValueError('Candidate grid is incomplete')
    return sorted(candidates, key=lambda c: (0 if c['method']=='adamw' and c['optimizer_kwargs']['matrix_lr']==3e-5 else 1,
                                             METHODS.index(c['method']), c['optimizer_kwargs']['matrix_lr']))


def required_free_bytes(plan, method, updates):
    if updates == 64: return 96*MIB
    group_bytes = plan.get('group_parameter_bytes')
    if type(group_bytes) is not int or group_bytes <= 0:
        raise ValueError('Recovery requires measured native group_parameter_bytes in the sealed plan')
    # Full0/1000/2000 + group500/1500, serialization temporary and metadata reserve.
    return max(370*MIB, 14*group_bytes + 64*MIB + (4*4128768*8 if method=='shampoo' else 0))


def trial_spec(plan, candidate, updates, decision_path=None):
    value = {'name': candidate['name'], 'method': candidate['method'], 'updates': updates,
             'optimizer_kwargs': candidate['optimizer_kwargs'], 'source_manifest': plan['source_manifest']}
    if updates == 64 and candidate['method']=='adamw' and candidate['optimizer_kwargs']['matrix_lr']==3e-5:
        value.update(parity_reference=plan['parity_reference'], parity_reference_sha256=plan['parity_reference_sha256'])
    if updates == 2000:
        value.update(qualification_path=str(decision_path), qualification_sha256=sha(decision_path))
    return value


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def inspect_trial(out, trial, *, control=False, exit_code=0):
    """Authenticate receipts; return eligibility without consulting development."""
    out = Path(out); completed_path = out/'completed.json'
    if not completed_path.is_file() or not (out/'launch.json').is_file():
        raise TechnicalFailure('Missing child launch/completion receipt')
    done, launch = read(completed_path), read(out/'launch.json')
    target = trial['updates']
    if (done.get('version') != RUNNER_VERSION or launch.get('version') != RUNNER_VERSION
            or launch.get('trial') != trial or done.get('method') != trial['method']
            or done.get('trial_name') != trial['name'] or done.get('neural_training_target') != target):
        raise TechnicalFailure('Child identity or bounded target changed')
    if (done.get('all_preservation_checks_passed') is not True or not done.get('preserved')
            or any(value is not True for value in done['preserved'].values())):
        raise TechnicalFailure('Child preservation failed or is missing')
    protected = launch.get('protected', {})
    if not protected or protected.get(str(Path(trial['source_manifest']).resolve())) != sha(trial['source_manifest']):
        raise TechnicalFailure('Child did not bind its executable source manifest')
    for path, expected in launch.get('protected', {}).items():
        if sha(path) != expected: raise TechnicalFailure('A child protected input changed')
    optimizer = done.get('optimizer', {})
    if (optimizer != launch.get('optimizer') or optimizer.get('method') != trial['method']
            or optimizer.get('adam') != {'lr':3e-5,'betas':[0.9,0.99],'eps':1e-8,'weight_decay':0.0}
            or optimizer.get('matrix',{}).get('lr') != trial['optimizer_kwargs']['matrix_lr']
            or optimizer.get('tensor_count') != 90):
        raise TechnicalFailure('Actual optimizer recipe differs from its requested trial')
    expected_sources = launch.get('source_ids', [])
    if len(expected_sources) != target*12 or len(set(expected_sources)) != len(expected_sources):
        raise TechnicalFailure('Invalid authenticated ordinary source prefix')
    if digest(expected_sources) != launch.get('source_ids_sha256'):
        raise TechnicalFailure('Source prefix hash differs')
    identity = {key: done.get(key) for key in ('initial_state_sha256', 'initial_rng_sha256',
        'initializer_artifact_sha256', 'source_plan_identity_sha256', 'training_probe_identity_sha256')}
    if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) for value in identity.values()):
        raise TechnicalFailure('Missing fresh initialization/probe identity')
    for key in ('initial_state_sha256', 'initial_rng_sha256', 'initializer_artifact_sha256', 'source_plan_identity_sha256'):
        if launch.get(key) != identity[key]: raise TechnicalFailure('Launch/completion identity differs')
    identity['ordinary_source_prefix_sha256'] = digest(expected_sources[:768])
    before = done.get('training_probe_before')
    if not isinstance(before, dict) or not before or not all(_finite_number(v) for v in before.values()):
        raise TechnicalFailure('Missing initial finite training probe')
    identity['training_probe_before_sha256'] = digest(before)
    rows = [json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()] if (out/'train.jsonl').is_file() else []
    seen = []; nonzero = 0
    for step, row in enumerate(rows, 1):
        ids = row.get('source_ids', [])
        if (row.get('step') != step or ids != expected_sources[(step-1)*12:step*12]
                or row.get('unique_sources') != step*12):
            raise TechnicalFailure('Child journal chronology/source ledger differs')
        checks = row.get('teacher_cache_checks', [])
        if len(checks)!=12 or any(c.get('allclose_original_tolerance') is not True for c in checks):
            raise TechnicalFailure('Teacher cache check failed or is missing')
        for key in ('q_caps_passed','q_full_primal_verified','q_kkt_passed'):
            if row.get(key) != 1: raise TechnicalFailure('Missing accepted full startup constraint certificate')
        if row.get('startup_anchor_after_passed') != 6 or row.get('startup_anchor_updates') != step:
            raise TechnicalFailure('Startup acceptance or counter failed')
        for key in ('total','waveform','mel','feature','q_accepted_displacement_norm','q_accepted_fraction'):
            if not _finite_number(row.get(key)): raise TechnicalFailure('Nonfinite or missing completed update scalar')
        if row['q_accepted_displacement_norm'] < 0 or not 0 <= row['q_accepted_fraction'] <= 1:
            raise TechnicalFailure('Invalid accepted displacement')
        if row.get('q_zero_displacement') not in (0,1): raise TechnicalFailure('Missing displacement certificate')
        if bool(row['q_zero_displacement']) != (row['q_accepted_displacement_norm']==0):
            raise TechnicalFailure('Zero-movement flag and actual displacement disagree')
        nonzero += row['q_zero_displacement']==0 and row['q_accepted_displacement_norm']>0
        seen.extend(ids)
    if (done.get('updates') != len(rows) or done.get('ordinary_unique_sources') != len(seen)
            or done.get('ordinary_source_prefix_sha256') != digest(seen)):
        raise TechnicalFailure('Completion and ordinary journal disagree')
    result = {'name': trial['name'], 'method': trial['method'], 'optimizer_kwargs': trial['optimizer_kwargs'],
              'completed_path': str(completed_path.resolve()), 'completed_sha256': sha(completed_path),
              'identity': identity, 'eligible': False, 'updates': len(rows), 'nonzero_updates': nonzero,
              'training_probe_total': None, 'parity_passed': done.get('parity', {}) is not None and done['parity'].get('passed') is True}
    if control and not result['parity_passed']: raise TechnicalFailure('Exact AdamW control parity failed')
    if done.get('complete') is not True:
        if done.get('failure_kind') != 'model_numeric':
            raise TechnicalFailure('Child failed without a certified model-numeric disposition')
        result['reason'] = 'model_numeric_failure'; return result
    if exit_code != 0 or len(rows) != target or seen != expected_sources or done.get('failure_category') is not None:
        raise TechnicalFailure('Complete child did not finish the exact target successfully')
    activity = done.get('optimizer_activity', {})
    if activity.get('matrix_parameters') != 9 or activity.get('matrix_steps') != [target]*9:
        raise TechnicalFailure('Nine genuine matrix counters were not authenticated')
    if trial['method']=='normuon' and activity.get('nor_neuron_state_shapes') != [[384,1]]*3+[[256,1]]*3+[[128,1]]*3:
        raise TechnicalFailure('NorMuon row adaptation did not engage')
    if trial['method']=='shampoo' and activity.get('shampoo_last_refresh_steps') != [target//10*10]*9:
        raise TechnicalFailure('Shampoo root refresh did not engage')
    after = done.get('training_probe_after', {})
    if not after or not all(_finite_number(v) for v in after.values()) or not _finite_number(after.get('total')):
        raise TechnicalFailure('Missing finite selection objective')
    if target == 2000:
        expected = {0:'full_optimizer',500:'group_only',1000:'full_optimizer',1500:'group_only',2000:'full_optimizer'}
        for step, kind in expected.items():
            receipt = read(out/f'checkpoint-step{step}.json')
            checkpoint = out/f'checkpoint-step{step}.pt'
            if (receipt != done.get('checkpoint_receipts',{}).get(str(step)) or receipt.get('step') != step
                    or receipt.get('source_count') != step*12 or receipt.get('snapshot_kind') != kind
                    or receipt.get('group_tensors') != 90 or receipt.get('sha256') != sha(checkpoint)
                    or receipt.get('bytes') != checkpoint.stat().st_size):
                raise TechnicalFailure('Checkpoint receipt/hash/counter differs')
    result.update(training_probe_total=after['total'], eligible=nonzero==target,
                  reason='qualified' if nonzero==target else 'zero_parameter_updates')
    return result


def select_candidates(plan, records, plan_sha256):
    """Pure deterministic selection; input contains no development metrics."""
    wanted = {c['name']: c for c in plan['candidates']}
    if len(records)!=12 or {r['name'] for r in records} != set(wanted):
        raise TechnicalFailure('All twelve candidates must have a recorded disposition')
    shared = records[0]['identity']
    for row in records:
        candidate = wanted[row['name']]
        if row['method']!=candidate['method'] or row['optimizer_kwargs']!=candidate['optimizer_kwargs'] or row['identity']!=shared:
            raise TechnicalFailure('Matched initialization, RNG, probe or source identity differs')
    control = next(r for r in records if r['method']=='adamw' and r['optimizer_kwargs']['matrix_lr']==3e-5)
    if not control['parity_passed']: raise TechnicalFailure('Exact AdamW control must pass before selection')
    selected = {}
    for method in METHODS:
        eligible = [r for r in records if r['method']==method and r['eligible']
                    and _finite_number(r['training_probe_total'])]
        if eligible:
            selected[method] = min(eligible, key=lambda r:(r['training_probe_total'],r['optimizer_kwargs']['matrix_lr'],r['name']))
        else: selected[method] = {'eligible':False, 'reason':'No numerically qualified nonzero64 candidate'}
    return {'version':DECISION_VERSION,'all_candidates_attempted':True,'control_parity_passed':True,
            'development_used_for_selection':False,'selection_metric':'training_probe_after.total',
            'selection_tie_break':'lowest matrix learning rate, then candidate name',
            'config_sha256':plan['config_sha256'],'source_manifest_sha256':plan['source_manifest_sha256'],
            'plan_sha256':plan_sha256,'matched_identity':shared,'candidates':records,'selected':selected}


def execute(plan, phase, candidate, target, decision_path, progress_path):
    root = Path(plan['root']); name = candidate['name']
    out = root/phase/name; tb = root/'tensorboard'/phase/name
    trial_path = root/'trials'/f'{phase}-{name}.json'; log_path = root/'logs'/f'{phase}-{name}.log'
    receipt_path = root/'launches'/f'{phase}-{name}.json'
    for path in (out,tb,trial_path,log_path,receipt_path):
        if path.exists(): raise FileExistsError(f'Preserved trial destination already exists: {path}')
    validate_plan(plan)
    needed = required_free_bytes(plan,candidate['method'],target); free = shutil.disk_usage(root).free
    if free < needed:
        return {'paused_storage':True,'name':name,'free_bytes':free,'required_bytes':needed,
                'reason':'Insufficient space for this fresh bounded arm and its retained snapshots; no files deleted'}
    spec = trial_spec(plan,candidate,target,decision_path); write_new(trial_path,spec)
    argv = [plan['python'], str(resolve_source(plan,'optimizer_comparison_runner.py')),
            '--config',plan['config'],'--trial',str(trial_path),'--out',str(out),'--tensorboard',str(tb)]
    env = os.environ.copy(); env['PYTHONPATH']=plan['pythonpath']; env['PYTHONUNBUFFERED']='1'
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with log_path.open('xb') as log:
        os.chmod(log_path,0o600)
        child = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                 env=env,start_new_session=True)
        identity = proc_identity(child.pid)
        write_new(receipt_path,{'version':VERSION,'launched_at':stamp(),'pid':child.pid,
            'start_ticks':None if identity is None else identity['start_ticks'],
            'already_exited':child.poll() is not None,'argv':argv,'trial_sha256':sha(trial_path)})
        while child.poll() is None:
            current = proc_identity(child.pid)
            if identity is not None and current is not None and current['start_ticks'] != identity['start_ticks']:
                raise TechnicalFailure('Child PID identity changed')
            journal = out/'train.jsonl'
            count = 0
            if journal.exists():
                with journal.open() as handle: count=sum(1 for line in handle if line.endswith('\n'))
            progress(progress_path,{'version':VERSION,'phase':phase,'status':'running','trial':name,
                'pid':child.pid,'start_ticks':None if identity is None else identity['start_ticks'],
                'completed_updates':count,'target_updates':target,'updated_at':stamp()})
            time.sleep(15)
        exit_code = child.wait()
    is_control = target==64 and candidate['method']=='adamw' and candidate['optimizer_kwargs']['matrix_lr']==3e-5
    return inspect_trial(out,spec,control=is_control,exit_code=exit_code)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--phase',choices=('qualification','recovery'),required=True)
    args=parser.parse_args(); plan=read(args.plan); candidates=validate_plan(plan)
    plan_hash=sha(args.plan); root=Path(plan['root']); root.mkdir(parents=True,exist_ok=True)
    progress_path=root/f'{args.phase}-progress.json'; summary_path=root/f'{args.phase}-controller-completed.json'
    launch_path=root/f'{args.phase}-controller-launch.json'
    for path in (progress_path,summary_path,launch_path):
        if path.exists(): raise FileExistsError('This controller phase has already been attempted; no automatic retries')
    for path in (root/'trials',root/'logs',root/'launches',root/args.phase,root/'tensorboard'/args.phase):
        path.mkdir(parents=True,exist_ok=True)
    decision_path=root/'qualification.json'; records=[]; status='running'
    write_new(launch_path,{'version':VERSION,'plan_sha256':plan_hash,'phase':args.phase,'pid':os.getpid(),
        'process':proc_identity(os.getpid()),'started_at':stamp(),'source_sha256':sha(__file__)})
    try:
        if args.phase=='recovery':
            decision=read(decision_path)
            if (sha(decision_path)!=(root/'qualification.sha256').read_text().strip()
                    or decision.get('plan_sha256')!=plan_hash
                    or select_candidates(plan,decision['candidates'],plan_hash)!=decision):
                raise TechnicalFailure('Sealed qualification decision or matched candidate record differs')
            for row in decision['candidates']:
                if sha(row['completed_path'])!=row['completed_sha256']:
                    raise TechnicalFailure('Qualification completion changed')
            candidates=[{'name':m,'method':m,'optimizer_kwargs':decision['selected'][m]['optimizer_kwargs']}
                        for m in METHODS if decision['selected'][m]['eligible']]
        elif decision_path.exists() or (root/'qualification.sha256').exists():
            raise FileExistsError('Never replace an existing qualification decision')
        for candidate in candidates:
            if sha(args.plan)!=plan_hash: raise TechnicalFailure('Plan changed during finite sequence')
            result=execute(plan,args.phase,candidate,64 if args.phase=='qualification' else 2000,
                           decision_path,progress_path)
            records.append(result)
            write_new(root/'launches'/f"{args.phase}-{candidate['name']}-result.json",result)
            if result.get('paused_storage'): status='paused_storage'; break
            if args.phase=='qualification' and len(records)>1 and result['identity']!=records[0]['identity']:
                raise TechnicalFailure('Matched source/probe/initialization identity differs')
            if args.phase=='recovery' and result['identity']!=decision['matched_identity']:
                raise TechnicalFailure('Fresh recovery differs from the qualified initialization/probe/source prefix')
        else:
            status='completed'
            if args.phase=='qualification':
                decision=select_candidates(plan,records,plan_hash); write_new(decision_path,decision)
                with (root/'qualification.sha256').open('x') as handle: handle.write(sha(decision_path)+'\n')
    except BaseException as exc:
        status='stopped_technical_failure'
        write_new(summary_path,{'version':VERSION,'status':status,'phase':args.phase,'plan_sha256':plan_hash,
            'finished_at':stamp(),'records':records,'failure_type':type(exc).__name__,'failure':str(exc),
            'automatic_retry':False,'automatic_promotion':False})
        raise
    write_new(summary_path,{'version':VERSION,'status':status,'phase':args.phase,'plan_sha256':plan_hash,
        'finished_at':stamp(),'records':records,'automatic_retry':False,'automatic_promotion':False})
    progress(progress_path,{'version':VERSION,'phase':args.phase,'status':status,'finished_at':stamp(),
                            'trials_finished':sum(not r.get('paused_storage',False) for r in records)})


if __name__=='__main__': main()
