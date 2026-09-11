"""Fresh G-plus-startup paired64 qualification and separately fresh corrected2k.

Historical step0 provides only the original empty Adam/RNG template. Every
student starts from the original teacher factory and four sealed native
initializer operators. Pilots are disposable; recovery never resumes them.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import json
import math
from pathlib import Path
import shutil
import time

import torch
import grail_startup_init as init
import reconstruction_b_recovery as b
import startup_corrected_anchor_update as corrected
import startup_corrected_anchor_pilot as corrected_pilot
import startup_retention_probe as startup

base,screen,replay,prior = b.base,b.screen,b.replay,b.prior
progressive,control,joint = b.progressive,b.control,b.joint
VERSION = 'audiovae2_grail_startup_recovery_v1'
PILOT_VERSION = 'audiovae2_grail_startup_pilot_v1'
OPERATOR_PATHS = b.OPERATOR_PATHS
ACCUMULATION,UPDATES = 12,2000
REVIEW_STEPS = (0,250,500,1000,1500,2000)
CHECKPOINT_STEPS = (0,1000,2000)
POLICY_PINS = {**corrected_pilot.PINNED_SOURCES,
    'startup_corrected_anchor_update.py':'9b4a39383b01c81e2d73bfb8b2a90fb054a042326845c63796a85a76014b491d',
    'startup_corrected_anchor_pilot.py':'88166b9a9f0a057561f3062e718ce228115fd104a1601b1350b0438796558f06',
    'grail_startup_init.py':'6bc3e3939af7276cfe7e393a0dd6bc1cf83bdcb6e8289de54fc88affd4c35152'}
INIT_FILES = ('grail-startup-native-operators.pt','grail-startup-receipt.json',
              'completed.json','launch.json','grail-startup-development-aggregate.json')


def install_operators(model,teacher,artifact,receipt,calibration_ids,*,artifact_sha256):
    """Validate the complete merged native state before any parameter write."""
    teacher_hash=control.state_hash(teacher.model.decoder)
    digest=screen.digest(list(calibration_ids))
    if (artifact.get('format')!=init.VERSION or artifact.get('variant')!=init.VARIANT
            or receipt.get('format')!=init.VERSION or receipt.get('variant')!=init.VARIANT
            or artifact.get('original_step0_sha256')!=b.STEP0_SHA
            or artifact.get('teacher_checkpoint_sha256')!=base.CHECKPOINT_SHA256
            or artifact.get('teacher_source_sha256')!=base.SOURCE_SHA256
            or artifact.get('teacher_state_sha256')!=teacher_hash or receipt.get('teacher_state_sha256')!=teacher_hash
            or artifact.get('selection')!=model.selections or receipt.get('selection')!=model.selections
            or len(calibration_ids)!=72 or len(set(calibration_ids))!=72
            or artifact.get('calibration_source_ids_sha256')!=digest or receipt.get('calibration_source_ids_sha256')!=digest
            or artifact.get('calibration_sources')!=72 or receipt.get('calibration_sources')!=72
            or artifact.get('ridge')!=init.constrained.RIDGE or artifact.get('automatic_promotion') is not False
            or artifact.get('g_operators_sha256')!=init.G_PINS['grail-native-operators.pt']
            or artifact.get('base_g_state_sha256')!=init.G_STATE_SHA
            or receipt.get('operators_sha256')!=artifact_sha256
            or receipt.get('changed_native_paths')!=list(OPERATOR_PATHS)
            or receipt.get('changed_relative_to_g')!=[OPERATOR_PATHS[-1]]
            or receipt.get('stage2_g_operators_preserved') is not True
            or receipt.get('native_fp32_constraints_passed') is not True
            or receipt.get('calibration_waveform_startup_passed')!=6
            or receipt.get('all9_residual_units_preserved') is not True
            or receipt.get('fresh_original_teacher_factory') is not True
            or receipt.get('original_checkpoint_weights_installed') is not False
            or receipt.get('unchanged_frozen_and_unselected_tensors') is not True
            or receipt.get('neural_training_updates')!=0 or receipt.get('extra_inference_modules')!=0
            or receipt.get('automatic_promotion') is not False
            or set(artifact.get('operators',{}))!=set(OPERATOR_PATHS)):
        raise ValueError('Sealed G-plus-startup provenance or native scope differs')
    current=model.decoder.state_dict(); pristine=b._state_hash(current)
    if artifact.get('base_selected_state_sha256')!=pristine or receipt.get('base_selected_state_sha256')!=pristine:
        raise ValueError('Operators require their original teacher factory basis')
    proposed=current.copy(); modules={}
    for path in OPERATOR_PATHS:
        module=model.decoder.get_submodule(path); expected=module.state_dict(); supplied=artifact['operators'][path]
        if supplied.keys()!=expected.keys():raise ValueError('Native operator state keys differ')
        for key,value in supplied.items():
            if (not torch.is_tensor(value) or value.shape!=expected[key].shape or value.dtype!=expected[key].dtype
                    or not torch.isfinite(value).all()):raise ValueError('Native operator tensor geometry or values differ')
            proposed[path+'.'+key]=value
        if path in OPERATOR_PATHS[:3] and b._state_hash(supplied)!=artifact.get('stage2_g_operator_hashes',{}).get(path):
            raise ValueError('A sealed G residual mixer differs')
        modules[path]=module
    wanted=b._state_hash(proposed)
    if wanted!=artifact.get('candidate_state_sha256') or wanted!=receipt.get('candidate_state_sha256'):
        raise ValueError('Native operators do not reconstruct the sealed complete initializer')
    objects=[(n,id(p),p.data_ptr()) for n,p in model.decoder.named_parameters()]
    for path,module in modules.items():module.load_state_dict(artifact['operators'][path],strict=True)
    if (objects!=[(n,id(p),p.data_ptr()) for n,p in model.decoder.named_parameters()]
            or control.state_hash(model.decoder)!=wanted or control.state_hash(teacher.model.decoder)!=teacher_hash):
        raise RuntimeError('Fresh installation changed objects, teacher or authenticated coefficients')
    return {'candidate_state_sha256':wanted,'base_selected_state_sha256':pristine,
            'parameter_objects_preserved':True,'old_group_state_loaded':False,'optimizer_updates':0,
            'only_four_native_operators_installed':True,'fit_repeated':False}


def restore_start(model,teacher,zero,artifact,receipt,calibration_ids,*,artifact_sha256):
    if (zero.get('cut_updates')!=0 or zero.get('global_updates')!=0 or zero.get('sources_seen')!=[]
            or zero.get('optimizer',{}).get('state')!={} or zero.get('selection')!=model.selections):
        raise ValueError('Require original zero-update selection and empty Adam template')
    installed=install_operators(model,teacher,artifact,receipt,calibration_ids,artifact_sha256=artifact_sha256)
    optimizer=progressive.fresh_optimizer(model,lr=3e-5)
    screen.restore_rng(zero['rng'])
    if (optimizer.state or not replay.compare_tree(optimizer.state_dict(),zero['optimizer'])['equal']
            or not replay.compare_tree(screen.rng_state(),zero['rng'])['equal']):
        raise RuntimeError('Original RNG or fresh Adam recipe did not reproduce')
    return optimizer,{**installed,'fresh_adam_matches_original':True,'original_step0_rng_restored_exactly':True}


def paired_qualification(ordinary,retention):
    """Mechanical eligibility plus measured trends; human review decides quality."""
    checks={}
    for label,row,method in (('ordinary',ordinary,'ordinary'),('retention',retention,'corrected')):
        checks[label+'_completed']=row.get('version')==PILOT_VERSION and row.get('method')==method and row.get('complete') is True
        checks[label+'_bounded']=row.get('updates')==64 and row.get('ordinary_unique_sources')==768 and row.get('checkpoint_written') is False
        checks[label+'_preserved']=row.get('all_preservation_checks_passed') is True
        checks[label+'_fresh']=row.get('installation',{}).get('old_group_state_loaded') is False and row.get('installation',{}).get('fresh_adam_matches_original') is True
        records=row.get('update_records',[])
        checks[label+'_records']=len(records)==64 and all(r.get('step')==i and
            all(not isinstance(v,float) or math.isfinite(v) for v in r.values()) for i,r in enumerate(records,1))
    for key in ('initial_state_sha256','initializer_artifact_sha256','config_sha256','ordinary_source_prefix_sha256',
                'source_plan_identity_sha256','initial_rng_sha256','policy_source_sha256','runner_source_sha256',
                'development_before','fixed_fitting_batch_objective_before','initial_calibration_startup','initial_development_startup'):
        checks[key+'_equal']=ordinary.get(key) is not None and ordinary.get(key)==retention.get(key)
    rows=retention.get('update_records',[])
    checks['all64_constraints']=len(rows)==64 and all(r.get('step')==i and r.get('q_constraints')==12
        and r.get('q_caps_passed')==r.get('q_normal_budget_passed')==r.get('q_full_primal_verified')==r.get('q_kkt_passed')==1
        and r.get('q_normal_full_primal_verified')==r.get('q_normal_kkt_passed')==1 and r.get('startup_anchor_after_passed')==6
        and r.get('q_normal_solves') in (0,1,2) for i,r in enumerate(rows,1))
    checks['last8_have_nonzero_learning']=len(rows)==64 and all(r.get('q_zero_displacement')==0
        and r.get('q_accepted_displacement_norm',0)>0 for r in rows[-8:])
    checks['calibration_end_passes']=retention.get('calibration_after',{}).get('passed')==6
    trends={}
    for key in ('mae','mel','group_mse','quiet_residual_rms_mean'):
        a=ordinary.get('development_after',{}).get('aggregate',{}).get(key)
        c=retention.get('development_after',{}).get('aggregate',{}).get(key)
        if isinstance(a,(int,float)) and isinstance(c,(int,float)) and math.isfinite(a) and math.isfinite(c):
            trends[key]={'ordinary':a,'corrected':c,'relative_change':(c/a-1) if a else None}
    return {'eligible_for_review':all(checks.values()),'checks':checks,'matched_quality':trends,
        'automatic_promotion':False,'human_quality_review_required':True,
        'last8_displacement_norm_sum':sum(r.get('q_accepted_displacement_norm',0) for r in rows[-8:]),
        'corrected_zero_updates':sum(r.get('q_zero_displacement',0) for r in rows)}


def storage_plan(parameter_bytes):
    if type(parameter_bytes)is not int or parameter_bytes<=0:raise ValueError('Positive native group size required')
    return {'group_tensor_bytes':parameter_bytes,'checkpoint_steps':list(CHECKPOINT_STEPS),
        'estimated_checkpoint_bytes':7*parameter_bytes+24*1024**2,
        'required_free_bytes':7*parameter_bytes+96*1024**2,
        'payload':'Native90 group tensors plus Adam moments, RNG and source ledger; no full decoder',
        'pilot_checkpoint_bytes':0}


def save_checkpoint(path,model,optimizer,step,source_ids,identity,selection,scored_samples):
    """Compact fixed milestones, never a pilot or earlier recovery start state."""
    path=Path(path)
    if path.exists() or path.with_suffix('.tmp').exists():raise FileExistsError('Checkpoint output already exists')
    if type(step)is not int or step not in CHECKPOINT_STEPS:raise ValueError('Only0/1000/2000 checkpoints are supported')
    ledger=identity['source_ids']
    if (len(ledger)!=24000 or len(set(ledger))!=24000 or screen.digest(ledger)!=identity['source_ids_sha256']
            or list(source_ids)!=ledger[:step*12] or len(source_ids)!=step*12 or selection!=identity['selection']):
        raise ValueError('Checkpoint source order or selected topology differs')
    params=base.parameters(model)
    if len(params)!=90 or (set(optimizer.state)!=set(params) if step else bool(optimizer.state)):
        raise ValueError('Checkpoint must cover all90 parameters and the exact Adam state')
    if any(float(s['step'])!=step or any(not torch.isfinite(s[k]).all() for k in ('exp_avg','exp_avg_sq')) for s in optimizer.state.values()):
        raise ValueError('Nonfinite or discontinuous Adam moments')
    for row in optimizer.param_groups:
        if row['lr']!=3e-5 or row['betas']!=(.9,.99) or row['eps']!=1e-8 or row['weight_decay']!=0 or [id(p) for p in row['params']]!=[id(p) for p in params]:
            raise ValueError('Original Adam recipe or ordered parameter scope changed')
    if type(scored_samples)is not int or (scored_samples<=0 if step else scored_samples!=0):
        raise ValueError('Invalid scored sample ledger')
    state={k:v.detach().cpu() for k,v in model.group_state_dict().items()}
    if len(state)!=90 or any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Invalid native group state')
    estimate=sum(v.numel()*v.element_size() for v in state.values())*(3 if step else 1)+8*1024**2
    if shutil.disk_usage(path.parent).free<estimate+32*1024**2:raise RuntimeError('Insufficient checkpoint storage headroom')
    payload={'format':VERSION,'group':state,'optimizer':optimizer.state_dict(),'rng':screen.rng_state(),
        'step':step,'cut_updates':step,'global_updates':step,'selection':selection,'sources_seen':list(source_ids),
        'scored_samples':scored_samples,'coefficients':dict(prior.COEFFICIENTS),'accumulation':12,
        'identity':identity,'teacher_source_sha256':base.SOURCE_SHA256,'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,
        'optimizer_reset_after_start':False,'automatic_next_cut':False,'automatic_promotion':False}
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
    return {'checkpoint_sha256':base.sha(path),'step':step,'source_count':len(source_ids),
        'bytes':path.stat().st_size,'optimizer_and_rng_saved':True,'optimizer_reset_after_start':False}


def read(path):return json.loads(Path(path).read_text())


def authenticate_inputs(config,data,pools):
    paths={key:Path(config[key]) for key in ('original_step0','original2000','recipe','schedule','manifest','source_plan')}
    if base.sha(paths['original_step0'])!=b.STEP0_SHA or base.sha(paths['original2000'])!=b.ORIGINAL2000_SHA:
        raise ValueError('Historical RNG/source templates differ from their original sealed bytes')
    # Read only named metadata; neither archive's group is accessed or installed.
    names=('format','cut_index','cut_updates','global_updates','sources_seen','selection','schedule_sha256',
           'coefficients','accumulation','teacher_source_sha256','teacher_checkpoint_sha256','identity','original_cut_identity')
    saved=torch.load(paths['original_step0'],map_location='cpu',weights_only=True,mmap=True)
    zero={k:saved[k] for k in (*names,'optimizer','rng') if k in saved};del saved
    saved=torch.load(paths['original2000'],map_location='cpu',weights_only=True,mmap=True)
    original={k:saved[k] for k in names if k in saved};del saved
    source_ids=b.validate_source_schedule(zero,original,data.source_ids)
    recipe=read(paths['recipe']);prior.validate_recipe(recipe)
    calibration_ids=[c['source_id'] for c in pools['calibration']]
    development_ids=[c['source_id'] for c in pools['development']]
    steps=prior.authenticate_schedule(read(paths['schedule']),manifest_sha=base.sha(paths['manifest']),
        source_plan_sha=base.sha(paths['source_plan']),calibration_ids=calibration_ids)
    if (zero['selection']!=steps[1] or zero['schedule_sha256']!=base.sha(paths['schedule'])
            or original.get('original_cut_identity')!=zero['identity']
            or zero['identity'].get('source_plan_identity_sha256')!=data.fresh.identity
            or zero['identity'].get('source_ids')!=list(source_ids[:12000])
            or zero['identity'].get('source_interval')!=[0,12000]
            or original['identity'].get('source_ids')!=list(source_ids[12000:])
            or original['identity'].get('source_interval')!=[12000,24000]
            or zero['identity'].get('recipe_sha256')!=base.sha(paths['recipe'])):
        raise ValueError('Historical source geometry, recipe or original cut identity differs')
    prior.validate_recipe(zero['identity']);prior.validate_recipe(original['identity'])
    old_done=read(paths['original2000'].parent/'completed.json')
    old_receipt=read(paths['original2000'].with_suffix('.json'))
    if (old_done.get('status')!='awaiting_review' or old_done.get('cut_updates')!=2000
            or old_done.get('last_checkpoint_sha256')!=b.ORIGINAL2000_SHA
            or old_done.get('frozen_state_preserved') is not True or old_done.get('original_files_preserved') is not True
            or old_receipt.get('checkpoint_sha256')!=b.ORIGINAL2000_SHA):
        raise ValueError('Original source ledger lacks its preserved completion')
    executed=[];extra=[]
    for folder,start,stop in ((paths['original_step0'].parent,1,1000),(paths['original2000'].parent,1001,2000)):
        journal=folder/'train.jsonl';records=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        if [r['step'] for r in records]!=list(range(start,stop+1)):raise ValueError('Original journal update sequence differs')
        executed.extend(sid for row in records for sid in row['source_ids']);extra.append(journal)
    if executed!=list(source_ids):raise ValueError('Original executed source prefix differs')
    folder=Path(config['initializer_dir']);pins=config['initializer_pins']
    if set(pins)!=set(INIT_FILES) or any(base.sha(folder/name)!=digest for name,digest in pins.items()):
        raise ValueError('A sealed G-plus-startup initialization artifact changed')
    artifact=torch.load(folder/INIT_FILES[0],map_location='cpu',weights_only=True)
    receipt=read(folder/INIT_FILES[1]);done=read(folder/'completed.json');launch=read(folder/'launch.json')
    reference=read(folder/'grail-startup-development-aggregate.json')
    if (done.get('version')!=init.VERSION or done.get('variant')!=init.VARIANT or done.get('complete') is not True
            or done.get('status')!='evaluated' or done.get('failure_category') is not None
            or done.get('files_preserved') is not True or not done.get('states_preserved') or not all(done['states_preserved'].values())
            or done.get('native_fp32_constraints_passed') is not True or done.get('stable_constraint_solution') is not True
            or done.get('calibration_sources')!=72 or done.get('development_sources')!=96 or done.get('neural_training_updates')!=0
            or done.get('optimizer_created') is not False or done.get('teacher_cache_comparisons')!=336
            or done.get('teacher_cache_all_pass') is not True
            or done.get('calibration_anchor_preparation',{}).get('comparisons')!=6
            or done.get('calibration_anchor_preparation',{}).get('all_pass') is not True
            or done.get('candidate_receipt')!=receipt or done.get('results',{}).get('grail_startup')!=reference
            or reference.get('calibration_startup',{}).get('passed')!=6
            or launch.get('calibration_source_ids_sha256')!=screen.digest(calibration_ids)
            or launch.get('development_source_ids_sha256')!=screen.digest(development_ids)
            or launch.get('original_step0_tensors_loaded') is not False
            or launch.get('g_pins')!=init.G_PINS or launch.get('selection')!=zero['selection']
            or artifact.get('candidate_state_sha256')!=config['initializer_state_sha256']
            or receipt.get('candidate_state_sha256')!=config['initializer_state_sha256']
            or receipt.get('operators_sha256')!=pins[INIT_FILES[0]]):
        raise ValueError('G-plus-startup initializer has not passed native and six-waveform qualification')
    if launch.get('torch')!=recipe['torch'] or launch.get('cudnn')!=recipe['cudnn']:
        raise ValueError('Initializer runtime differs from the original recovery recipe')
    extra += [paths['original2000'].parent/'completed.json',paths['original2000'].with_suffix('.json')]
    extra += [folder/name for name in INIT_FILES]
    extra += [Path(config['assets'])/'audio_vae_v2.py',Path(config['assets'])/'audiovae.pth']
    protected={**zero['identity']['protected'],**original['identity']['protected'],
               **{str(p.resolve()):base.sha(p) for p in (*paths.values(),*extra)}}
    if any(base.sha(p)!=h for p,h in protected.items()):raise ValueError('A preserved original source or recipe input changed')
    return zero,artifact,receipt,source_ids,recipe,reference,protected


def build(config):
    import fresh_training_data
    base.policy();manifest,pools,_=base.load_data(Path(config['manifest']))
    if len(pools['calibration'])!=72 or len(pools['development'])!=96:raise ValueError('Fixed calibration/development pools differ')
    fresh=fresh_training_data.FreshTrainingData(Path(config['source_plan']),Path(config['manifest']),pools,Path(config['shards']))
    data=prior.SourceStream(pools,fresh)
    zero,artifact,receipt,source_ids,recipe,reference,protected=authenticate_inputs(config,data,pools)
    if str(torch.__version__)!=recipe['torch'] or torch.backends.cudnn.version()!=recipe['cudnn']:
        raise ValueError('Qualified Torch/cuDNN runtime differs')
    process=control.gpu_idle_snapshot();assets=Path(config['assets'])
    teacher=base.FrozenAudioVAE2.from_files(assets/'audio_vae_v2.py',assets/'audiovae.pth',device='cuda')
    model=progressive.initialize_from_teacher(teacher.model.decoder,artifact['selection'])
    optimizer,installation=restore_start(model,teacher,zero,artifact,receipt,[c['source_id'] for c in pools['calibration']],
        artifact_sha256=config['initializer_pins'][INIT_FILES[0]])
    if len(base.parameters(model))!=90 or any(p.requires_grad for p in teacher.model.parameters()):
        raise RuntimeError('Original teacher freezing or native90-parameter topology differs')
    return manifest,pools,data,zero,teacher,model,optimizer,source_ids,reference,protected,installation,process


def validate_review(config,ordinary_path,retention_path,decision_path):
    ordinary,retention,decision=read(ordinary_path),read(retention_path),read(decision_path)
    report=paired_qualification(ordinary,retention)
    current={'runner_source_sha256':base.sha(__file__),'policy_source_sha256':base.sha(corrected.__file__),
        'initializer_artifact_sha256':config['initializer_pins'][INIT_FILES[0]],
        'initial_state_sha256':config['initializer_state_sha256']}
    if (not report['eligible_for_review'] or any(ordinary.get(k)!=v for k,v in current.items())
            or decision.get('decision')!='qualified'
            or decision.get('target_updates')!=2000 or decision.get('fresh_start') is not True
            or decision.get('ordinary_pilot_sha256')!=base.sha(ordinary_path)
            or decision.get('corrected_pilot_sha256')!=base.sha(retention_path)
            or decision.get('initializer_artifact_sha256')!=config['initializer_pins'][INIT_FILES[0]]
            or decision.get('initializer_state_sha256')!=config['initializer_state_sha256']
            or decision.get('runner_source_sha256')!=base.sha(__file__)
            or decision.get('policy_source_sha256')!=base.sha(corrected.__file__)
            or decision.get('source_prefix_sha256')!=ordinary['ordinary_source_prefix_sha256']
            or decision.get('config_sha256')!=ordinary['config_sha256']
            or not isinstance(decision.get('quality_assessment'),str) or not decision['quality_assessment'].strip()):
        raise ValueError('A bound paired-pilot quality review is required before fresh2000 recovery')
    return report,{'ordinary_pilot_sha256':base.sha(ordinary_path),'corrected_pilot_sha256':base.sha(retention_path),
                   'decision_sha256':base.sha(decision_path)}


def rng_digest(state):
    return screen.digest({**state,'torch':state['torch'].tolist(),'cuda':[t.tolist() for t in state['cuda']]})


def main():
    import fresh_training_data
    import grail_startup_monitor as monitor_module
    from unified_monitor import UnifiedMonitor
    from joint_recovery_gates_v2 import summarize_regions
    from compare_accumulation_v1 import numeric_reference_check
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=('pilot','recovery'),required=True)
    parser.add_argument('--method',choices=('ordinary','corrected'),required=True)
    parser.add_argument('--ordinary-pilot',type=Path);parser.add_argument('--corrected-pilot',type=Path)
    parser.add_argument('--qualification',type=Path);parser.add_argument('--tensorboard',type=Path)
    args=parser.parse_args();config=read(args.config);pilot=args.mode=='pilot';target=64 if pilot else UPDATES
    if args.out.exists():raise FileExistsError('Use a new independent run directory')
    if not pilot and args.method!='corrected':raise ValueError('Only the qualified corrected arm has a2000-update entry point')
    logdir=args.tensorboard or args.out/'tensorboard'
    if logdir.exists():raise FileExistsError('Use new independent TensorBoard events')
    retained=[Path(config[k]).resolve() for k in ('initializer_dir','assets','shards')]
    retained += [Path(config[k]).resolve().parent for k in ('original_step0','original2000','recipe')]
    if any(new.is_relative_to(old) or old.is_relative_to(new) for new in (args.out.resolve(),logdir.resolve()) for old in retained):
        raise ValueError('New results would overlap a preserved input directory')
    qualification=None;review_identity={}
    if not pilot:
        if not all((args.ordinary_pilot,args.corrected_pilot,args.qualification)):
            raise ValueError('Both matched pilot receipts and their explicit quality review are required')
        qualification,review_identity=validate_review(config,args.ordinary_pilot,args.corrected_pilot,args.qualification)
        if read(args.ordinary_pilot).get('config_sha256')!=base.sha(args.config):
            raise ValueError('Recovery config differs from its paired pilot start')
    modules=(init,b,corrected,corrected_pilot,corrected.complete,corrected.complete.projection,
        corrected.correction,corrected.correction.signed,corrected.correction.zero,
        corrected.anchor,corrected.q,corrected.anchor.warmup,corrected.grid,corrected.correction.pilot)
    for module in modules:
        path=Path(module.__file__)
        if path.name in POLICY_PINS and base.sha(path)!=POLICY_PINS[path.name]:
            raise ValueError('A frozen qualified policy source changed')
    built=build(config)
    manifest,pools,data,zero,teacher,model,optimizer,source_ids,reference,protected,installation,process=built
    paths=[args.config,Path(__file__),Path(monitor_module.__file__),*(Path(m.__file__) for m in modules),
           *(Path(m.__file__) for m in (prior,base,screen,replay,joint,progressive,control,startup,startup.helper,startup.quiet,fresh_training_data))]
    if not pilot:paths += [args.ordinary_pilot,args.corrected_pilot,args.qualification]
    protected.update({str(p.resolve()):base.sha(p) for p in paths})
    initial={k:v.detach().clone() for k,v in model.group_state_dict().items()}
    initial_hash=control.state_hash(model.decoder);initial_optimizer=copy.deepcopy(optimizer.state_dict())
    initial_rng=screen.rng_state();teacher_hash=control.state_hash(teacher.model)
    frozen=screen.frozen_versions(model);teacher_versions=b.resume.continuation.teacher_versions(teacher)
    plan=storage_plan(sum(v.numel()*v.element_size() for v in initial.values()))
    ancestor=args.out.parent
    while not ancestor.exists():ancestor=ancestor.parent
    free=shutil.disk_usage(ancestor).free
    if free<(48*1024**2 if pilot else plan['required_free_bytes']):raise RuntimeError('Insufficient bounded output/checkpoint storage')
    args.out.mkdir(parents=True)
    metadata={r['source_id']:r for r in manifest['splits']['development']['rows']}
    monitor=None if pilot else monitor_module.RecoveryMonitor(logdir,source_metadata=metadata)
    screen.restore_rng(zero['rng'])
    identity={'version':PILOT_VERSION if pilot else VERSION,'mode':args.mode,'method':args.method,
        'initialization':'Fresh original teacher factory plus four sealed G-plus-startup operators',
        'initializer_artifact_sha256':config['initializer_pins'][INIT_FILES[0]],
        'initial_state_sha256':initial_hash,'initial_rng_sha256':rng_digest(zero['rng']),
        'original_step0_sha256':b.STEP0_SHA,'original2000_sha256':b.ORIGINAL2000_SHA,
        'config_sha256':base.sha(args.config),'runner_source_sha256':base.sha(__file__),
        'policy_source_sha256':base.sha(corrected.__file__),'selection':model.selections,
        'source_plan_identity_sha256':data.fresh.identity,'source_ids':list(source_ids),
        'source_ids_sha256':screen.digest(list(source_ids)),'schedule_sha256':zero['schedule_sha256'],
        'coefficients':dict(prior.COEFFICIENTS),'learning_rate':3e-5,'optimizer_betas':[.9,.99],
        'optimizer_eps':1e-8,'weight_decay':0,'execution_batch_size':1,'gradient_accumulation':12,
        'trainable_stages':[2,3,4],'original_teacher_forever':True,'original_checkpoint_weights_installed':False,
        'installation':installation,'target_updates':target,'source_interval':[0,target*12],
        'review_steps':list((0,64) if pilot else REVIEW_STEPS),'checkpoint_steps':[] if pilot else list(CHECKPOINT_STEPS),
        'checkpoint_written':False,'protected':protected,'torch':str(torch.__version__),
        'cudnn':torch.backends.cudnn.version(),'backend':replay.backend_state(),'process_snapshot':process,
        'storage_plan':plan,'disk_free_before':free,'qualification':review_identity,
        'new_loss_weights':False,'new_inference_operations':0,'automatic_next_cut':False,'automatic_promotion':False,
        'pilot_state_resumed':False,'recurring_calibration_sources':6 if args.method=='corrected' else 0,
        'current_batch_quiet_constraints':False,'development_used_for_updates':False,
        'normal_fraction_semantics':'Base projected fraction; added normal motion and actual movement reported separately'}
    base.write_json(args.out/'launch.json',identity)
    report={k:identity[k] for k in ('version','method','initial_state_sha256','initializer_artifact_sha256','config_sha256',
        'initial_rng_sha256','source_plan_identity_sha256','runner_source_sha256','policy_source_sha256','installation')}
    report.update(complete=False,updates=0,ordinary_unique_sources=0,checkpoint_written=False,
        failure_category=None,reviews=[],update_records=[],neural_training_target=target,
        automatic_promotion=False,pilot_state_resumed=False,qualification=review_identity)
    seen=[];samples=0;last_receipt=None;status='running';started=time.monotonic()
    elapsed_updates=0.;elapsed_validation=0.;ordinary_seconds=0.;auxiliary_seconds=0.;milestones={};warmed=set()
    totals={'ordinary_adam_updates':0,'constrained_updates':0,'normal_accepts':0,'normal_solves':0,
        'constraint_zero_displacements':0,'extended_grid_fallbacks':0,'canonical_score_forwards':0,
        'constraint_gradient_forwards':0,'accepted_displacement_norm_sum':0.,'maximum_normal_over_projected':0.}
    old_backtrack,old_score=corrected.q.backtrack,corrected.q.score_entries
    old_q_grid,old_anchor_grid=corrected.q.FRACTIONS,corrected.anchor.FRACTIONS
    initial_compare=None;anchors=[];development=[];common=None
    def assert_frozen():
        if screen.frozen_versions(model)!=frozen or b.resume.continuation.teacher_versions(teacher)!=teacher_versions:
            raise RuntimeError('A frozen teacher or outer student tensor changed')
    def evaluate(step):
        nonlocal elapsed_validation
        tick=time.monotonic()
        with replay.diagnostic_state_guard(model,teacher,optimizer):
            result=joint.evaluate_review(UnifiedMonitor(None,{},metadata),model,teacher,pools['development'],common,summarize=summarize_regions)
        elapsed_validation+=time.monotonic()-tick;assert_frozen()
        if not pilot:base.write_json(args.out/f'development-step{step}.json',result)
        summary=init.summarize_quality(result);milestones[step]=summary
        return result,summary
    try:
        with corrected.grid.extended_grid():
            anchors,anchor_cache=startup.startup_panel(teacher,pools['calibration'],6)
            development,development_cache=startup.startup_panel(teacher,pools['development'],13)
            updater=corrected.StartupAnchorUpdate(model,teacher,anchors,optimizer)
            report['initialization']=updater.receipt
            report['startup_cache_checks']={'calibration':anchor_cache,'development':development_cache}
            common=base.objective()
            report['initial_calibration_startup']=startup.score_panel(model,anchors)
            report['initial_development_startup']=startup.score_panel(model,development)
            prior.warm_student(model,teacher,pools['development'],warmed,optimizer)
            full,summary=evaluate(0)
            report['development_before']=summary
            expected={k:reference[k] for k in summary}
            initial_compare=numeric_reference_check(summary,expected)
            cal_compare=numeric_reference_check(report['initial_calibration_startup'],reference['calibration_startup'])
            if (not initial_compare['passed'] or not cal_compare['passed']
                    or control.state_hash(model.decoder)!=initial_hash
                    or not replay.compare_tree(optimizer.state_dict(),zero['optimizer'])['equal']):
                raise RuntimeError('Fresh initializer metrics/state or empty Adam did not reproduce')
            first=data.take(0,12);prior.warm_student(model,teacher,first,warmed,optimizer)
            report['fixed_fitting_batch_objective_before']=startup.ordinary_objective(model,teacher,first,common)
            screen.restore_rng(zero['rng'])
            if monitor is not None:
                monitor.log_validation(full,0)
                last_receipt=save_checkpoint(args.out/'checkpoint-step0.pt',model,optimizer,0,[],identity,model.selections,0)
                last_receipt['quality']=full['aggregate'];base.write_json(args.out/'checkpoint-step0.json',last_receipt)
            for step in range(1,target+1):
                crops=data.take((step-1)*12,12);ids=[c['source_id'] for c in crops]
                if ids!=list(source_ids[(step-1)*12:step*12]) or set(ids)&set(seen):raise RuntimeError('Ordinary source prefix changed or repeated')
                prior.warm_student(model,teacher,crops,warmed,optimizer)
                tick=time.monotonic()
                update=updater.perform_update if args.method=='corrected' else prior.perform_update
                values,checks=update(model,teacher,crops,common,optimizer,diagnostics=True)
                duration=time.monotonic()-tick;elapsed_updates+=duration
                ordinary_seconds+=values.get('q_ordinary_update_seconds',duration)
                auxiliary_seconds+=values.get('q_auxiliary_seconds',0.)
                if len(checks)!=12 or not all(c['allclose_original_tolerance'] for c in checks):raise RuntimeError('Ordinary teacher target cache changed')
                if set(optimizer.state)!=set(base.parameters(model)) or any(float(s['step'])!=step for s in optimizer.state.values()):
                    raise RuntimeError('Original all90 Adam counters differ from completed updates')
                assert_frozen();seen.extend(ids);samples+=sum(c['samples'] for c in checks)
                totals['ordinary_adam_updates']+=1
                if args.method=='corrected':
                    totals['constrained_updates']+=1
                    for destination,key in (('normal_accepts','q_normal_accepted'),('normal_solves','q_normal_solves'),
                        ('constraint_zero_displacements','q_zero_displacement'),('extended_grid_fallbacks','q_extended_grid_fallback'),
                        ('canonical_score_forwards','q_canonical_score_forwards'),('constraint_gradient_forwards','q_constraint_gradient_forwards'),
                        ('accepted_displacement_norm_sum','q_accepted_displacement_norm')):
                        totals[destination]+=values[key]
                    totals['maximum_normal_over_projected']=max(totals['maximum_normal_over_projected'],values['q_normal_norm_over_projected'])
                row={k:v for k,v in values.items() if (k.startswith(('q_','startup_anchor_')) or k in (*prior.COEFFICIENTS,'total'))
                     and (v is None or isinstance(v,(int,float,bool)))}
                row.update(step=step,step_seconds=duration,elapsed_seconds=time.monotonic()-started,
                    unique_sources=len(seen),audio_hours=samples/48000/3600,training_update_seconds=elapsed_updates,
                    validation_seconds=elapsed_validation,waiting_seconds=0.)
                if pilot:report['update_records'].append(row)
                with (args.out/'train.jsonl').open('a') as handle:
                    handle.write(json.dumps({**row,'source_ids':ids,'teacher_cache_checks':checks},allow_nan=False)+'\n')
                if monitor is not None:monitor.log_training(row,step)
                report['updates']=step
                if pilot and (step<=8 or step in (16,32,64)):
                    report['reviews'].append({'step':step,'calibration':startup.score_panel(model,anchors),
                        'development':startup.score_panel(model,development)})
                if step==1 or step%25==0:base.event('grail_startup_progress',step=step,target=target,method=args.method,total=values['total'])
                if not pilot and step in REVIEW_STEPS:
                    full,summary=evaluate(step)
                    summary['calibration_startup']=startup.score_panel(model,anchors)
                    summary['development_startup']=startup.score_panel(model,development)
                    monitor.log_validation(full,step);data.assert_unchanged()
                    if step in CHECKPOINT_STEPS:
                        last_receipt=save_checkpoint(args.out/f'checkpoint-step{step}.pt',model,optimizer,step,seen,identity,model.selections,samples)
                        last_receipt['quality']=full['aggregate'];base.write_json(args.out/f'checkpoint-step{step}.json',last_receipt)
                    base.write_json(args.out/f'review-step{step}.json',summary)
                    base.event('grail_startup_review',step=step,aggregate=summary['aggregate'],automatic_next_cut=False)
            if pilot:
                report['fixed_fitting_batch_objective_after']=startup.ordinary_objective(model,teacher,first,common)
                _,report['development_after']=evaluate(64)
            else:report['development_after']=milestones[2000]
            report['calibration_after']=startup.score_panel(model,anchors)
            report['development_startup_after']=startup.score_panel(model,development)
            status='completed_pilot' if pilot else 'awaiting_review';report['complete']=True
    except BaseException as exc:
        report['failure_category']=type(exc).__name__;status='failed';raise
    finally:
        if monitor is not None:monitor.close()
        if pilot:
            model.load_group_state_dict(initial);optimizer.load_state_dict(initial_optimizer)
            optimizer.zero_grad(set_to_none=True);screen.restore_rng(initial_rng)
        data.assert_unchanged()
        preserved={'teacher':control.state_hash(teacher.model)==teacher_hash,
            'frozen_outer':screen.frozen_versions(model)==frozen,
            'files':all(base.sha(p)==h for p,h in protected.items()),
            'score_backtrack_globals':corrected.q.backtrack is old_backtrack and corrected.q.score_entries is old_score,
            'fraction_globals':corrected.q.FRACTIONS is old_q_grid and corrected.anchor.FRACTIONS is old_anchor_grid}
        if pilot:preserved.update(initial_model=control.state_hash(model.decoder)==initial_hash,
            initial_optimizer=replay.compare_tree(optimizer.state_dict(),initial_optimizer)['equal'],
            initial_rng=replay.compare_tree(screen.rng_state(),initial_rng)['equal'])
        report.update(status=status,preserved=preserved,all_preservation_checks_passed=all(preserved.values()),
            ordinary_unique_sources=len(seen),ordinary_source_prefix_sha256=screen.digest(seen),scored_samples=samples,
            checkpoint_written=not pilot and last_receipt is not None,
            last_checkpoint_sha256=last_receipt['checkpoint_sha256'] if last_receipt else None,
            milestone_quality={str(k):v for k,v in milestones.items()},
            training_update_seconds=elapsed_updates,ordinary_update_seconds=ordinary_seconds,
            constraint_auxiliary_seconds=auxiliary_seconds,validation_seconds=elapsed_validation,
            elapsed_seconds=time.monotonic()-started,recurring_calibration_sources=6 if args.method=='corrected' else 0,
            anchor_update_participations=6*len(seen)//12 if args.method=='corrected' else 0,
            current_batch_quiet_constraints=False,initial_parity_passed=bool(initial_compare and initial_compare['passed']))
        report['update_totals']=totals
        report['complete']=report['complete'] and all(preserved.values()) and len(seen)==target*12
        base.write_json(args.out/'completed.json',report)
        if not all(preserved.values()):raise RuntimeError('Fresh run failed original-state or runtime preservation')


if __name__=='__main__':main()
