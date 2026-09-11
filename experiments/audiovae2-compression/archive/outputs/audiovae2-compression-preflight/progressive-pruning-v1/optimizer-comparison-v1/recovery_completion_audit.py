"""One completed optimizer arm: CPU receipts and raw checkpoint tensors only.

Run with the qualified Python and --method adamw|muon|normuon|shampoo.
No decoder, optimizer, forward pass or CUDA context is constructed. Serialized
states are checked before any cast; group-only snapshots are never resumable.
Source identities/tensors stay local. Only allowlisted aggregate evidence leaves
this process. The method's audit output is created exclusively and never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

# This must precede the torch import, even if another GPU arm is running.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import torch

ROOT = Path('/dev/shm/fast-audiovae-optimizer-comparison-20260911-v1')
VERSION = 'audiovae2_optimizer_recovery_completion_audit_v1'
PLAN_SHA = 'c219369ccc09efe9e7b0d7843f778a4d0e38f7d1d6dcac3efa6669e58bf195a7'
PINS = {
    'optimizer_comparison_controller.py':'d5cc30b0cee8124b7e4de292bd339f35468ac3845a8f616f09023829a79b64b8',
    'optimizer_comparison_runner.py':'6ac8954ac0bc5f4cd6a18dd169e37190c3624d731a8809f324b608095e6f6147',
    'recovery_optimizers.py':'2bc934f067765a1294c1d7e6afe1f1e39596f3994732b9f2a9e69ef9940f5ab7',
    'source-manifest.json':'a1a396c990dd09ed31752312563a851f7a3a8cd9da239244ed3fa73d6013b085',
    'config.json':'2697da9ae19094dde15a107b3bb4f756a65d9868861011fe121cd721183cf282',
    'qualification.json':'25ac5da9f67015086962b9249a87a74dde74f1df02837218af28ebbc1756b809',
}
METHODS = ('adamw','muon','normuon','shampoo')
SNAPSHOTS = {0:'full_optimizer',500:'group_only',1000:'full_optimizer',1500:'group_only',2000:'full_optimizer'}
REVIEWS = (0,250,500,1000,1500,2000)
REGIONS = ('all_quiet','near_silence','near_startup_first20ms','near_after800ms',
           'quiet_nonzero_reference','source_zero_20to40ms','source_zero_after40ms')
REGION_METRICS = ('windows','samples','failed','amplitude_failed','teacher_rms','output_rms',
                  'residual_rms','centered_residual_rms','output_limit_excess_rms')
CATEGORIES = ('passed','residual_only','amplitude_only','both')
COEFFICIENTS = {'waveform':1.,'mel':0.0006674012905982311,'feature':0.009304078923434964}
NATIVE_PATHS = tuple(f'model.3.block.{i}.block.3' for i in (2,3,4))+('model.4.block.1',)
QUALITY_METRICS = ('sources','samples','nonquiet_samples','mae','mse','waveform_nrmse','mel','group_mse','group_nrmse',
                   'nonquiet_cosine_mean','quiet_residual_rms_mean','quiet_windows','quiet_failed_windows','peak_abs_max','overshoot_samples')
STARTUP_METRICS = ('windows','passed','residual_only_failed','amplitude_only_failed','both_failed','samples',
                  'residual_max_excess','amplitude_max_excess','residual_rms','student_rms','teacher_rms',
                  'residual_dc_rms','residual_ac_rms','residual_mean')
STEP0_SHA = 'bd9a09c1ab5e86fce8f5ba1f435565c9dfff76b83c32d413d037a54d0df5942f'


class AuditFailure(RuntimeError):
    """Messages are fixed categories, never raw exceptions or source identifiers."""


def require(condition, category):
    if not condition:
        raise AuditFailure(category)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda _: (_ for _ in ()).throw(AuditFailure('nonfinite_json')))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def load_pinned_module(name):
    path = ROOT/(name+'.py')
    require(sha(path)==PINS[path.name],'executable_hash')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_load(path, expected_sha):
    require(sha(path)==expected_sha,'tensor_archive_hash_before_load')
    require(not torch.cuda.is_initialized(),'unexpected_cuda_context')
    return torch.load(path,map_location='cpu',weights_only=True,mmap=True)


def rng_digest(state):
    require(set(state)=={'torch','cuda','python','numpy'},'rng_schema')
    tensors = [state['torch'],*state['cuda']]
    require(isinstance(state['cuda'],list) and len(state['cuda'])==1,'rng_cuda_state_count')
    require(all(torch.is_tensor(t) and t.device.type=='cpu' and t.dtype==torch.uint8
                and t.ndim==1 and t.numel()>0 for t in tensors),'rng_tensor_schema')
    require(isinstance(state['python'],tuple) and len(state['python'])==3
            and isinstance(state['numpy'],list) and len(state['numpy'])==5,'rng_host_schema')
    return digest({**state,'torch':state['torch'].tolist(),'cuda':[t.tolist() for t in state['cuda']]})


def protection_record(row, step):
    fractions = tuple(2.**(-power) for power in range(11))+(0.,)
    flags = ('q_full_primal_verified','q_kkt_passed','q_caps_passed',
             'q_normal_full_primal_verified','q_normal_kkt_passed','q_normal_budget_passed')
    return (row.get('step')==step and row.get('q_constraints')==12
        and row.get('q_constrained_sources')==row.get('q_constraint_gradient_sources')==6
        and all(row.get(k)==1 for k in flags)
        and row.get('q_normal_correction_enabled')==1 and row.get('startup_anchor_after_passed')==6
        and row.get('startup_anchor_updates')==step and row.get('q_accepted_fraction') in fractions
        and row.get('q_base_fraction')==row.get('q_accepted_fraction')
        and row.get('q_normal_solves') in (0,1,2)
        and row.get('q_normal_accepted') in (0,1)
        and row.get('q_base_fraction_has_normal_correction')==row.get('q_normal_accepted')
        and (not row.get('q_normal_accepted') or row.get('q_base_fraction')==1.)
        and row.get('q_normal_budget_fraction')==.25
        and 0<=row.get('q_normal_accepted_norm',math.inf)<=row.get('q_normal_budget',-1.)
        and row.get('q_canonical_score_forwards')==row.get('startup_anchor_checks')==6*row.get('q_canonical_score_calls',-1)
        and row.get('q_constraint_gradient_forwards')==6+row.get('q_normal_gradient_sources',-1)
        and row.get('q_normal_gradient_sources')==6*row.get('q_normal_solves',-1)
        and row.get('q_actual_norm_over_optimizer')==row.get('q_actual_norm_over_adam')
        and all(not isinstance(v,float) or math.isfinite(v) for v in row.values()))


def geometry(group, cfg):
    names, shapes = cfg['canonical_names'],cfg['canonical_shapes']
    require(len(names)==len(set(names))==len(shapes)==len(group)==90 and set(group)==set(names),'group_topology')
    for name,shape in zip(names,shapes):
        value = group[name]
        require(torch.is_tensor(value) and value.device.type=='cpu' and value.dtype==torch.float32
                and list(value.shape)==shape and bool(torch.isfinite(value).all()),'group_tensor_geometry_or_finiteness')
    return {name:(tuple(group[name].shape),str(group[name].dtype)) for name in names}


def optimizer_state(raw, cfg, group, step, ro, baseline_groups):
    """Validate serialized tensors directly; never hide a bad dtype by loading/casting."""
    names = cfg['canonical_names']; matrix = set(ro.MATRIX_NAMES)
    require(set(raw)==({'state','param_groups'} if cfg['method']=='adamw' else {'state','param_groups','recovery_config'}),'optimizer_archive_schema')
    require(set(cfg['matrix_names'])==matrix and len(matrix)==9,'matrix_allowlist')
    require(cfg['tensor_count']==90 and cfg['matrix_tensor_count']==(0 if cfg['method']=='adamw' else 9),'optimizer_tensor_scope')
    for stage,width in ((3,384),(4,256),(5,128)):
        for unit in (2,3,4):
            require(tuple(group[f'model.{stage}.block.{unit}.block.3.weight_v'].shape)==(width,width,1),'matrix_native_shape')
    groups = raw['param_groups']
    if cfg['method']=='adamw' and cfg['matrix']['lr']==cfg['adam']['lr']:
        ordered = [names]
    else:
        ordered = [[n for n in names if n not in matrix],[n for n in names if n in matrix]]
    require(len(groups)==len(ordered),'optimizer_group_partition')
    require([g['params'] for g in groups]==[list(range(sum(map(len,ordered[:i])),sum(map(len,ordered[:i+1]))))
            for i in range(len(ordered))],'optimizer_serialized_parameter_mapping')
    if baseline_groups is not None:
        require(groups==baseline_groups,'optimizer_groups_changed_since_fresh_start')
    if cfg['method']!='adamw':
        require(raw.get('recovery_config')==cfg,'mixed_optimizer_config')
        require({k:v for k,v in groups[1].items() if k!='params'}=={'algorithm':cfg['method'],**cfg['matrix']},'matrix_group_recipe')
    for i,g in enumerate(groups):
        if cfg['method']!='adamw' and i==1:
            continue
        rate = cfg['matrix']['lr'] if i==1 else cfg['adam']['lr']
        require(g['lr']==rate and tuple(g['betas'])==(.9,.99) and g['eps']==1e-8
                and g['weight_decay']==0 and all(g.get(k) is False for k in ('amsgrad','maximize','capturable','differentiable'))
                and g.get('foreach') is None and g.get('fused') is None,'native_adam_recipe')
        if cfg['method']!='adamw':require(g.get('algorithm')=='adamw','native_adam_group_kind')
    states = raw['state']; mapping = {sid:name for g,ns in zip(groups,ordered) for sid,name in zip(g['params'],ns)}
    require(not states if step==0 else set(states)==set(mapping) and len(states)==90,'optimizer_state_count')
    matrix_states = 0; refreshes = []
    for sid,state in states.items():
        name = mapping[sid]; p = group[name]
        require(ro._step(state['step'])==step,'optimizer_counter')
        tensor = lambda key,shape,dtype: ro._tensor_state(state[key],shape,dtype,torch.device('cpu'),key)
        if cfg['method']=='adamw' or name not in matrix:
            require(set(state)=={'step','exp_avg','exp_avg_sq'},'adam_state_schema')
            for key in ('exp_avg','exp_avg_sq'):tensor(key,p.shape,torch.float32)
            require(bool((state['exp_avg_sq']>=0).all()),'adam_negative_second_moment')
        elif cfg['method'] in ('muon','normuon'):
            matrix_states += 1
            require(set(state)=={'step','momentum'}|({'variance_neuron'} if cfg['method']=='normuon' else set()),'matrix_momentum_schema')
            tensor('momentum',p.shape[:2],torch.float32)
            if cfg['method']=='normuon':
                tensor('variance_neuron',(p.shape[0],1),torch.float32)
                require(bool((state['variance_neuron']>=0).all()),'neuron_negative_second_moment')
        else:
            matrix_states += 1; m,n = p.shape[:2]; mc = cfg['matrix']
            expected = step//mc['precondition_frequency']*mc['precondition_frequency'] if step>=mc['start_preconditioning_step'] else 0
            require(set(state)==ro.DOUBLE_STATE|{'step','last_refresh'},'shampoo_state_schema')
            require(type(state['last_refresh']) is int and state['last_refresh']==expected,'shampoo_refresh_counter')
            refreshes.append(expected)
            for key,shape in (('gradient_ema',(m,n)),('graft_second',(m,n)),('factor_left',(m,m)),('factor_right',(n,n)),
                              ('inverse_left',(m,m) if expected else (0,0)),('inverse_right',(n,n) if expected else (0,0))):
                tensor(key,shape,torch.float64)
            require(bool((state['graft_second']>=0).all()),'shampoo_negative_graft_moment')
    return {'states':len(states),'matrix_method_states':matrix_states,'counters':step,
            'shampoo_refreshed_states':len(refreshes),'shampoo_last_refresh':min(refreshes) if refreshes else None,
            'finite_raw_states':True,'raw_dtype_and_mapping_verified':True},groups


def aggregate_quality(summary):
    scalar = lambda v: v is None or finite(v)
    require(set(summary['regions'])==set(REGIONS),'quality_region_scope')
    require(all(k in summary['aggregate'] and scalar(summary['aggregate'][k]) for k in QUALITY_METRICS),'quality_aggregate_schema')
    result = {'aggregate':{k:summary['aggregate'][k] for k in QUALITY_METRICS}, 'regions':{},
              'teacher_window_identity_sha256':summary['teacher_window_identity_sha256']}
    for name in REGIONS:
        row = summary['regions'][name]
        require(all(k in row and scalar(row[k]) for k in REGION_METRICS),'quality_region_schema')
        result['regions'][name] = {k:row[k] for k in REGION_METRICS}
        require(all(finite(row['failure_categories'][k]) for k in CATEGORIES),'quality_failure_category_schema')
        result['regions'][name]['failure_categories'] = {k:row['failure_categories'][k] for k in CATEGORIES}
    for key in ('calibration_startup','development_startup'):
        require(all(k in summary[key] and scalar(summary[key][k]) for k in STARTUP_METRICS),'startup_aggregate_schema')
        result[key] = {k:summary[key][k] for k in STARTUP_METRICS}
    for key in ('active_rms_ratio','whistle_rms_ratio'):
        require(scalar(summary.get(key)),'quality_ratio_schema')
        result[key] = summary.get(key)
    return result


def audit(method):
    checks = {}
    def checked(name, condition):
        require(condition,name)
        checks[name] = True

    checked('sealed_plan',sha(ROOT/'plan.json')==PLAN_SHA)
    checked('sealed_executables_config_and_qualification',all(sha(ROOT/name)==pin for name,pin in PINS.items()))
    c = load_pinned_module('optimizer_comparison_controller')
    ro = load_pinned_module('recovery_optimizers')
    plan = read(ROOT/'plan.json'); candidates = c.validate_plan(plan)
    checked('fixed_root_and_runtime',Path(plan['root'])==ROOT
            and Path(sys.executable).resolve()==Path(plan['python']).resolve())
    # Runtime identity is also checked against each launch below; the pinned plan
    # supplies the qualified interpreter, not a reason to initialize CUDA here.
    config = read(ROOT/'config.json'); decision = read(ROOT/'qualification.json')
    checked('qualification_seal',(ROOT/'qualification.sha256').read_text().strip()==PINS['qualification.json'])
    checked('complete_twelve_candidate_decision',len(candidates)==12
            and c.select_candidates(plan,decision['candidates'],PLAN_SHA)==decision)
    qualification_done = read(ROOT/'qualification-controller-completed.json')
    checked('qualification_controller_receipt',qualification_done['status']=='completed'
            and qualification_done['records']==decision['candidates'])
    checked('all_qualification_completions_unchanged',all(sha(r['completed_path'])==r['completed_sha256'] for r in decision['candidates']))
    selected = decision['selected'][method]
    checked('selected_method_eligible',selected.get('eligible') is True)
    candidate_done = read(selected['completed_path'])
    out = ROOT/'recovery'/method
    trial_path = ROOT/'trials'/f'recovery-{method}.json'; trial = read(trial_path)
    expected_trial = c.trial_spec(plan,{'name':method,'method':method,'optimizer_kwargs':selected['optimizer_kwargs']},2000,ROOT/'qualification.json')
    checked('fresh_selected_trial',trial==expected_trial)
    record = c.inspect_trial(out,trial,exit_code=0)
    done, launch = read(out/'completed.json'),read(out/'launch.json')
    checked('completed_method_scope',done.get('complete') is True and done.get('status')=='awaiting_comparison'
            and done.get('failure_category') is None and done.get('updates')==done.get('neural_training_target')==2000)
    checked('controller_method_record',read(ROOT/'launches'/f'recovery-{method}-result.json')==record)
    child = read(ROOT/'launches'/f'recovery-{method}.json')
    process = c.proc_identity(child['pid'])
    checked('method_process_exited',process is None or process['state']=='Z' or str(process['start_ticks'])!=str(child['start_ticks']))
    checked('child_trial_bytes',child['trial_sha256']==sha(trial_path))
    checked('matched_qualified_identity',record['identity']==decision['matched_identity'])
    checked('runtime_matches_launch',str(torch.__version__)==launch['torch'])
    checked('fresh_initialization_receipts',done.get('pilot_state_resumed') is False
            and launch.get('pilot_state_resumed') is False and launch.get('automatic_next_cut') is False
            and launch.get('automatic_promotion') is False and done.get('automatic_promotion') is False
            and launch['initial_state_sha256']==config['initializer_state_sha256']
            and launch['installation']['old_group_state_loaded'] is False
            and launch['installation']['only_four_native_operators_installed'] is True
            and launch['installation']['optimizer_updates']==0
            and launch['installation']['original_step0_rng_restored_exactly'] is True)
    checked('original_objective_and_execution',launch['coefficients']==COEFFICIENTS
            and launch['execution_batch_size']==1 and launch['accumulation']==12)
    cfg = launch['optimizer']
    checked('qualified_optimizer_recipe',cfg==done['optimizer']==candidate_done['optimizer']
            and cfg['method']==method and cfg['matrix']['lr']==selected['optimizer_kwargs']['matrix_lr'])
    preserved = done['preserved']
    checked('required_preservation_flags',all(preserved.get(k) is True for k in
            ('teacher','frozen_outer','files','fraction_globals','score_globals'))
            and done['all_preservation_checks_passed'] is True)
    protected = launch['protected']
    checked('protected_config_and_trial',protected.get(str((ROOT/'config.json').resolve()))==PINS['config.json']
            and protected.get(str(trial_path.resolve()))==sha(trial_path))

    rows = [read_line for line in (out/'train.jsonl').read_text().splitlines() if (read_line:=json.loads(line))]
    checked('all_2000_normal_constraint_certificates',len(rows)==2000
            and all(protection_record(row,step) for step,row in enumerate(rows,1)))
    ids = [sid for row in rows for sid in row['source_ids']]
    checked('complete_24000_unique_order',len(ids)==len(set(ids))==24000 and ids==launch['source_ids']
            and digest(ids)==launch['source_ids_sha256']==done['ordinary_source_prefix_sha256']
            and digest(ids[:768])==decision['matched_identity']['ordinary_source_prefix_sha256'])
    original_ids = []
    for key,start,stop in (('original_step0',1,1000),('original2000',1001,2000)):
        journal = Path(config[key]).parent/'train.jsonl'
        require(protected.get(str(journal.resolve()))==sha(journal),'original_journal_hash')
        historical = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        require([r['step'] for r in historical]==list(range(start,stop+1))
                and all(len(r['source_ids'])==12 for r in historical),'original_journal_sequence')
        original_ids.extend(sid for row in historical for sid in row['source_ids'])
        del historical
    checked('same_original_24000_executed_sources',original_ids==ids)
    del original_ids
    prefix_samples = [0]
    for row in rows:
        cache = row['teacher_cache_checks']
        require(len(cache)==12 and all(v['allclose_original_tolerance'] is True
                and type(v['samples']) is int and v['samples']>0 for v in cache),'teacher_cache_records')
        prefix_samples.append(prefix_samples[-1]+sum(v['samples'] for v in cache))
    checked('sample_and_anchor_exposure',prefix_samples[-1]==done['scored_samples']
            and done['recurring_calibration_sources']==6 and done['anchor_update_participations']==12000)
    checked('training_timing_reconciles',sum(r['step_seconds'] for r in rows)==done['training_update_seconds']
            and rows[-1]['training_update_seconds']==done['training_update_seconds']
            and done['validation_seconds']>=rows[-1]['validation_seconds'])

    # Independently reconstruct the expected group from the authenticated
    # pristine archive plus four native initializer operators. This is tensor
    # comparison only: neither archive is installed into a decoder or optimizer.
    original_zero = cpu_load(config['original_step0'],STEP0_SHA)
    artifact_path = Path(config['initializer_dir'])/'grail-startup-native-operators.pt'
    artifact = cpu_load(artifact_path,config['initializer_pins'][artifact_path.name])
    checked('initializer_archive_identity',set(artifact['operators'])==set(NATIVE_PATHS)
            and artifact['candidate_state_sha256']==launch['initial_state_sha256']
            and artifact['selection']==launch['selection']==original_zero['selection']
            and artifact['original_step0_sha256']==STEP0_SHA)
    checked('frozen_teacher_identity',launch['teacher_source_sha256']==artifact['teacher_source_sha256']==original_zero['teacher_source_sha256']
            and launch['teacher_checkpoint_sha256']==artifact['teacher_checkpoint_sha256']==original_zero['teacher_checkpoint_sha256'])
    checked('original_rng_identity',rng_digest(original_zero['rng'])==launch['initial_rng_sha256'])
    checked('original_source_plan_identity',original_zero['identity']['source_plan_identity_sha256']==launch['source_plan_identity_sha256']
            and original_zero['identity']['source_ids']==ids[:12000]
            and original_zero['identity']['source_interval']==[0,12000])
    expected_group = dict(original_zero['group'])
    for path,state in artifact['operators'].items():
        for name,value in state.items():
            require(path+'.'+name in expected_group,'initializer_native_scope')
            expected_group[path+'.'+name] = value
    snapshots = {}; expected_geometry = None; baseline_groups = None
    for step,kind in SNAPSHOTS.items():
        path = out/f'checkpoint-step{step}.pt'; receipt = read(path.with_suffix('.json'))
        checked(f'checkpoint_{step}_receipt',receipt==done['checkpoint_receipts'][str(step)]
                and receipt['step']==step and receipt['source_count']==step*12
                and receipt['snapshot_kind']==kind and receipt['group_tensors']==90
                and receipt['bytes']==path.stat().st_size)
        ck = cpu_load(path,receipt['sha256'])
        require(ck['format']==c.RUNNER_VERSION and ck['step']==step and ck['snapshot_kind']==kind,'checkpoint_format_or_step')
        require(ck['identity']==launch and ck['selection']==launch['selection'] and ck['optimizer_config']==cfg
                and ck['sources_seen']==ids[:step*12] and ck['scored_samples']==prefix_samples[step]
                and ck['automatic_next_cut'] is False and ck['automatic_promotion'] is False,'checkpoint_identity_or_ledger')
        current_geometry = geometry(ck['group'],cfg)
        if expected_geometry is None:expected_geometry = current_geometry
        require(current_geometry==expected_geometry,'checkpoint_geometry_changed')
        if step==0:
            checked('step0_exact_native_initialization',set(ck['group'])==set(expected_group)
                    and all(torch.equal(ck['group'][n],expected_group[n]) for n in expected_group))
        item = {'sha256':receipt['sha256'],'bytes':receipt['bytes'],'group_tensors':90,'kind':kind,
                'source_count':step*12,'scored_samples':prefix_samples[step],'geometry_finite_and_stable':True}
        if kind=='group_only':
            require('optimizer' not in ck and 'rng' not in ck,'group_only_contains_resume_state')
            item.update(optimizer_states_saved=False,rng_saved=False)
        else:
            state_report,groups = optimizer_state(ck['optimizer'],cfg,ck['group'],step,ro,baseline_groups)
            if baseline_groups is None:baseline_groups = groups
            rh = rng_digest(ck['rng'])
            if step==0:checked('step0_rng_matches_original',rh==launch['initial_rng_sha256'])
            item.update(optimizer=state_report,rng_sha256=rh,optimizer_states_saved=True,rng_saved=True)
        item['all_pass'] = True; snapshots[str(step)] = item
        del ck
    del original_zero,artifact,expected_group
    checked('all_five_checkpoints',len(snapshots)==5)

    quality = {}; panel = None; window_hash = None; region_support = None
    for step in REVIEWS:
        summary = read(out/f'review-step{step}.json')
        require(summary==done['milestone_quality'][str(step)],'review_completion_equality')
        full = read(out/f'development-step{step}.json')
        source_panel = [r['source_id'] for r in full['rows']]
        require(len(source_panel)==len(set(source_panel))==96 and full['aggregate']['sources']==96,'fixed96_review_scope')
        require(all(full['aggregate'][k]==summary['aggregate'][k] for k in QUALITY_METRICS),'review_aggregate_equality')
        require(set(full['quiet_regions']['regions'])==set(REGIONS),'full_quiet_region_scope')
        for name in REGIONS:
            require(all(full['quiet_regions']['regions'][name][k]==summary['regions'][name][k] for k in REGION_METRICS),'review_region_equality')
            require(all(full['quiet_regions']['regions'][name]['failure_categories'][k]==summary['regions'][name]['failure_categories'][k]
                        for k in CATEGORIES),'review_failure_category_equality')
        require(full['quiet_regions']['window_identity_sha256']==summary['teacher_window_identity_sha256'],'review_window_hash')
        support = {name:tuple(summary['regions'][name][k] for k in ('windows','samples','teacher_rms')) for name in REGIONS}
        if panel is None:panel,window_hash,region_support = source_panel,summary['teacher_window_identity_sha256'],support
        require(source_panel==panel and window_hash==summary['teacher_window_identity_sha256'] and support==region_support,'review_fixed_windows_changed')
        require(summary['calibration_startup']['windows']==6 and summary['development_startup']['windows']==13,'startup_panel_scope')
        quality[str(step)] = aggregate_quality(summary)
        del full
    checked('fixed_review_panel_and_regions',set(quality)==set(map(str,REVIEWS)))
    checked('immutable_inputs_rehashed_after_audit',all(sha(path)==h for path,h in protected.items()))
    checked('cpu_only_no_cuda_context',os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized())
    totals = {name:sum(r[key] for r in rows) for name,key in (
        ('normal_solves','q_normal_solves'),('normal_accepts','q_normal_accepted'),('zero_displacements','q_zero_displacement'),
        ('extended_grid_fallbacks','q_extended_grid_fallback'),('canonical_score_forwards','q_canonical_score_forwards'),
        ('constraint_gradient_forwards','q_constraint_gradient_forwards'))}
    totals.update(ordinary_updates=2000,nonzero_updates=record['nonzero_updates'],ordinary_unique_sources=24000,
        teacher_cache_comparisons=24000,anchor_update_participations=12000,
        accepted_displacement_norm_sum=sum(r['q_accepted_displacement_norm'] for r in rows))
    return {'version':VERSION,'method':method,'utc':datetime.now(timezone.utc).isoformat(),'all_pass':True,
        'checks':checks,'plan_sha256':PLAN_SHA,'qualification_sha256':PINS['qualification.json'],
        'completion_sha256':sha(out/'completed.json'),'launch_sha256':sha(out/'launch.json'),
        'trial_sha256':sha(trial_path),'source_ledger_sha256':digest(ids),'initial_state_sha256':launch['initial_state_sha256'],
        'source_manifest_sha256':PINS['source-manifest.json'],'auditor_sha256':sha(__file__),
        'optimizer_method':method,'matrix_learning_rate':cfg['matrix']['lr'],
        'protected_files_rehashed':len(protected),'checkpoints':snapshots,'quality':quality,'totals':totals,
        'scored_samples':prefix_samples[-1],'scored_audio_hours':prefix_samples[-1]/48000/3600,
        'timing':{'training_update_seconds':done['training_update_seconds'],
            'ordinary_update_seconds':sum(r['q_ordinary_update_seconds'] for r in rows),
            'constraint_auxiliary_seconds':sum(r['q_auxiliary_seconds'] for r in rows),
            'validation_seconds':done['validation_seconds'],'elapsed_seconds':done['elapsed_seconds']},
        'no_model_inference':True,'no_optimizer_constructed':True,'device':'cpu','automatic_promotion':False,
        'audit_scope':'Raw saved group/optimizer/RNG tensors and receipts; frozen outer/teacher preservation and cached-target agreement use the authenticated run receipts and unchanged input bytes, without rerunning networks.',
        'interpretation':'Mechanical completion and state integrity; no requirement of improved quality or nonzero movement on every update.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS,required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    output = ROOT/f'recovery-{args.method}-completion-audit-aggregate.json'
    if output.exists():
        print(json.dumps({'all_pass':False,'status':'audit_output_already_exists','method':args.method}))
        return 2
    # An unfinished method is not a failed completed-run audit; do not consume
    # its exclusive output name while the other optimizer processes continue.
    complete_path = ROOT/'recovery'/args.method/'completed.json'
    result_path = ROOT/'launches'/f'recovery-{args.method}-result.json'
    if not complete_path.is_file() or not result_path.is_file():
        print(json.dumps({'all_pass':False,'status':'not_ready','method':args.method}))
        return 2
    try:
        result = audit(args.method)
    except Exception as exc:
        result = {'version':VERSION,'method':args.method,'utc':datetime.now(timezone.utc).isoformat(),
            'all_pass':False,'failure_category':str(exc) if isinstance(exc,AuditFailure) else type(exc).__name__,
            'no_model_inference':True,'device':'cpu','automatic_promotion':False}
    with output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('quality','checkpoints')},allow_nan=False))
    return 0 if result['all_pass'] else 1


if __name__=='__main__':
    sys.exit(main())
