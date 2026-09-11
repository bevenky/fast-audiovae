"""Bounded recovery of the frozen reconstruction-aware B initialization.

The original step0 is restored, then its four native fitted operators are
installed. No fit is repeated, and no adapted checkpoint initializes training.
The first24,000 original sources are consumed once in their original order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
import progressive_train as prior
import progressive_model as progressive
import identical_teacher_control as control

base, screen, replay, resume, joint = prior.base, prior.screen, prior.replay, prior.resume, prior.joint
VERSION = 'audiovae2_reconstruction_b_recovery_v1'
INIT_VERSION = 'audiovae2_reconstruction_aware_initialization_v1'
STEP0_SHA = 'bd9a09c1ab5e86fce8f5ba1f435565c9dfff76b83c32d413d037a54d0df5942f'
ORIGINAL2000_SHA = 'a7b6c5ee850c80061c073dd42ab7a019ce96d55b0b9691b77c9721ea0b14602f'
STEP5000_SHA = '4dd64e0d165ab23aa56e5c9f0e0fe0d281e510dfb3db8d6cd863a2be0640fe47'
B_SHA = '7281436098cf01ac3aee09056651bdad4b7ddc112b82a22ad380edc2073866d7'
B_STATE_SHA = 'b6e8950402780c72951ccbae4da009d99e27e9bc0c1cf7ea2a161569d8c9cf22'
OPERATOR_PATHS = tuple(f'model.3.block.{i}.block.3' for i in (2,3,4))+('model.4.block.1',)
ACCUMULATION, UPDATES = 12, 2000
REVIEW_STEPS = (0,250,500,1000,1500,2000)
CHECKPOINT_STEPS = (0,1000,2000)


def _state_hash(state):
    checksum = hashlib.sha256()
    for name,value in state.items():
        checksum.update(name.encode()); checksum.update(str(value.dtype).encode())
        checksum.update(str(tuple(value.shape)).encode())
        checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def install_b_operators(model, artifact, receipt, *, artifact_sha256, step0_sha256, calibration_ids):
    """Validate the whole artifact before any write; retain native parameter objects.

    Shape-generic for CPU fixtures. Main independently enforces production384/256.
    The receipt's full decoder hash is checked against the proposed merged state
    before loading, and against the actual native decoder after loading.
    """
    if (artifact.get('format') != INIT_VERSION or artifact.get('variant') != 'B'
            or artifact.get('base_step0_sha256') != step0_sha256
            or artifact.get('selection') != model.selections
            or artifact.get('fit_source_ids') != list(calibration_ids)
            or len(calibration_ids) != 72 or len(set(calibration_ids)) != 72
            or artifact.get('ridge') != 1e-6
            or receipt.get('operators_sha256') != artifact_sha256
            or receipt.get('changed_native_paths') != list(OPERATOR_PATHS)
            or receipt.get('unchanged_frozen_and_unselected_tensors') is not True
            or receipt.get('all9_residual_units_preserved') is not True
            or receipt.get('neural_training_updates') != 0
            or receipt.get('extra_inference_modules') != 0
            or receipt.get('automatic_promotion') is not False
            or set(artifact.get('operators',{})) != set(OPERATOR_PATHS)):
        raise ValueError('B initialization provenance, source partition or native operator scope differs')
    state = model.decoder.state_dict()
    pristine = receipt.get('original_step0_pristine',{})
    if pristine.get('passed') is not True or pristine.get('student_state_sha256') != _state_hash(state):
        raise ValueError('B operators must be installed on the original unadapted step0')
    proposed = state.copy(); modules = {}
    for path in OPERATOR_PATHS:
        module = model.decoder.get_submodule(path); supplied = artifact['operators'][path]
        expected = module.state_dict()
        if set(supplied) != set(expected): raise ValueError('Native operator state keys differ: '+path)
        for key,value in supplied.items():
            if (not isinstance(value,torch.Tensor) or value.shape != expected[key].shape
                    or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
                raise ValueError('Native operator tensor geometry/dtype/finiteness differs: '+path+'.'+key)
            proposed[path+'.'+key] = value
        modules[path] = module
    expected_hash = _state_hash(proposed)
    if expected_hash != receipt.get('candidate_state_sha256'):
        raise ValueError('B operator coefficients do not reconstruct the authenticated candidate')
    parameters = [(name,id(p),p.data_ptr()) for name,p in model.decoder.named_parameters()]
    for path,module in modules.items(): module.load_state_dict(artifact['operators'][path],strict=True)
    if (parameters != [(n,id(p),p.data_ptr()) for n,p in model.decoder.named_parameters()]
            or control.state_hash(model.decoder) != expected_hash):
        raise RuntimeError('Native B installation changed parameter topology or coefficients')
    return {'candidate_state_sha256':expected_hash,'only_four_native_operators_installed':True,
            'parameter_objects_preserved':True,'fit_repeated':False,'optimizer_updates':0}


def validate_source_schedule(step0, original2000, source_ids):
    """Authenticate the exact historical prefix; this is diagnostic reuse across runs."""
    selected = tuple(source_ids[:UPDATES*ACCUMULATION])
    if (step0.get('format') != prior.VERSION or step0.get('cut_index') != 1
            or step0.get('cut_updates') != 0 or step0.get('global_updates') != 0
            or step0.get('sources_seen') != [] or step0.get('optimizer',{}).get('state') != {}
            or original2000.get('format') != 'audiovae2_progressive_same_cut_continuation_v1'
            or original2000.get('cut_index') != 1 or original2000.get('cut_updates') != UPDATES
            or original2000.get('global_updates') != UPDATES
            or len(selected) != 24000 or len(set(selected)) != 24000
            or list(selected) != original2000.get('sources_seen')
            or step0.get('selection') != original2000.get('selection')
            or step0.get('schedule_sha256') != original2000.get('schedule_sha256')):
        raise ValueError('Original first24,000 source order, width, start state or lineage differs')
    for payload in (step0,original2000):
        if (payload.get('coefficients') != prior.COEFFICIENTS or payload.get('accumulation') != ACCUMULATION
                or payload.get('teacher_source_sha256') != base.SOURCE_SHA256
                or payload.get('teacher_checkpoint_sha256') != base.CHECKPOINT_SHA256):
            raise ValueError('Original source ledger belongs to a different training recipe/teacher')
    return selected


def restore_b_start(model, teacher, zero, artifact, receipt, calibration_ids, *, artifact_sha256=B_SHA):
    model.load_group_state_dict(zero['group'])
    if not replay.compare_tree(dict(model.group_state_dict()),dict(zero['group']))['equal']:
        raise RuntimeError('Original step0 group did not restore exactly')
    teacher_hash = control.state_hash(teacher.model.decoder)
    if receipt['original_step0_pristine']['teacher_state_sha256'] != teacher_hash:
        raise ValueError('B calibration used a different frozen teacher')
    installed = install_b_operators(model,artifact,receipt,artifact_sha256=artifact_sha256,
                                   step0_sha256=STEP0_SHA,calibration_ids=calibration_ids)
    optimizer = progressive.fresh_optimizer(model,lr=3e-5)
    screen.restore_rng(zero['rng'])
    if (optimizer.state or not replay.compare_tree(optimizer.state_dict(),zero['optimizer'])['equal']
            or not replay.compare_tree(screen.rng_state(),zero['rng'])['equal']
            or control.state_hash(teacher.model.decoder) != teacher_hash):
        raise RuntimeError('Original fresh AdamW, RNG or teacher did not restore exactly')
    return optimizer,{**installed,'original_step0_rng_restored_exactly':True,'fresh_adam_matches_original':True}


def save_checkpoint(path,model,optimizer,step,source_ids,identity,selection,scored_samples):
    path = Path(path)
    if path.exists() or path.with_suffix('.tmp').exists(): raise FileExistsError(path)
    if type(step) is not int or step not in CHECKPOINT_STEPS:
        raise ValueError('Only step0/1000/2000 checkpoint states are authorized')
    expected = identity['source_ids'][:step*ACCUMULATION]
    if (len(identity['source_ids']) != 24000 or len(set(identity['source_ids'])) != 24000
            or screen.digest(identity['source_ids']) != identity['source_ids_sha256']
            or list(source_ids) != expected or len(source_ids) != step*ACCUMULATION
            or selection != identity['selection']):
        raise ValueError('B checkpoint source ledger or selection differs')
    params = base.parameters(model)
    if (not params or (set(optimizer.state) != set(params) if step else bool(optimizer.state))
            or any(float(s['step']) != step for s in optimizer.state.values())):
        raise ValueError('B checkpoint Adam counters/scope differ from uninterrupted training')
    for row in optimizer.param_groups:
        if (row['lr'] != 3e-5 or row['betas'] != (.9,.99) or row['eps'] != 1e-8 or row['weight_decay'] != 0
                or set(row['params']) != set(params)):
            raise ValueError('B checkpoint optimizer settings changed')
    if type(scored_samples) is not int or (scored_samples <= 0 if step else scored_samples != 0):
        raise ValueError('Missing exact scored audio accounting')
    payload = {'format':VERSION,'group':{k:v.detach().cpu() for k,v in model.group_state_dict().items()},
        'optimizer':optimizer.state_dict(),'rng':screen.rng_state(),'step':step,'cut_index':1,'cut_updates':step,
        'global_updates':step,'selection':selection,'schedule_sha256':identity['schedule_sha256'],
        'sources_seen':list(source_ids),'scored_samples':scored_samples,'coefficients':dict(prior.COEFFICIENTS),
        'accumulation':ACCUMULATION,'teacher_source_sha256':base.SOURCE_SHA256,
        'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,'identity':identity,
        'optimizer_reset_after_start':False,'automatic_next_cut':False,'automatic_promotion':False}
    # Group+two Adam moments, without a full decoder copy. Allow serialization overhead.
    estimate = sum(t.numel()*t.element_size() for t in payload['group'].values())*(3 if step else 1)+8*1024**2
    if shutil.disk_usage(path.parent).free < estimate+32*1024**2:
        raise RuntimeError('Insufficient disk headroom for the bounded group/optimizer checkpoint')
    tmp = path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path)
    return {'checkpoint_sha256':base.sha(path),'step':step,'sources_seen':len(source_ids),
            'optimizer_and_rng_saved':True,'optimizer_reset_after_start':False,'bytes':path.stat().st_size}


# Reuse the unchanged original singleton/pooled objective, cache observer and warmup.
perform_update = prior.perform_update


def authenticate_inputs(args,data,pools):
    """Bind B, original RNG/ledger, native architecture and all saved references."""
    for path,expected in ((args.step0,STEP0_SHA),(args.original2000,ORIGINAL2000_SHA),
                          (args.preserved5000,STEP5000_SHA),
                          (args.b_dir/'variant-B-native-operators.pt',B_SHA)):
        if base.sha(path) != expected: raise ValueError('Unexpected preserved artifact: '+str(path))
    zero = torch.load(args.step0,map_location='cpu',weights_only=True,mmap=True)
    original = torch.load(args.original2000,map_location='cpu',weights_only=True,mmap=True)
    source_ids = validate_source_schedule(zero,original,data.source_ids)
    recipe = json.loads(args.recipe.read_text()); prior.validate_recipe(recipe)
    calibration_ids = [c['source_id'] for c in pools['calibration']]
    development_ids = [c['source_id'] for c in pools['development']]
    steps = prior.authenticate_schedule(json.loads(args.schedule.read_text()),manifest_sha=base.sha(args.manifest),
        source_plan_sha=base.sha(args.source_plan),calibration_ids=calibration_ids)
    if (zero['selection'] != steps[1] or zero['schedule_sha256'] != base.sha(args.schedule)
            or original.get('original_cut_identity') != zero['identity']
            or zero['identity'].get('source_plan_identity_sha256') != data.fresh.identity
            or zero['identity'].get('source_ids') != list(source_ids[:12000])
            or zero['identity'].get('source_interval') != [0,12000]
            or zero['identity'].get('recipe_sha256') != base.sha(args.recipe)
            or original['identity'].get('source_ids') != list(source_ids[12000:])
            or original['identity'].get('source_interval') != [12000,24000]
            or zero['identity'].get('precision') != 'FP32 TF32 disabled'):
        raise ValueError('Original schedule, source geometry provenance or optimizer recipe changed')
    prior.validate_recipe(zero['identity'])
    prior.validate_recipe(original['identity'])
    completion = json.loads((args.original2000.parent/'completed.json').read_text())
    checkpoint_receipt = json.loads(args.original2000.with_suffix('.json').read_text())
    if (completion.get('status') != 'awaiting_review' or completion.get('cut_updates') != 2000
            or completion.get('last_checkpoint_sha256') != ORIGINAL2000_SHA
            or completion.get('frozen_state_preserved') is not True
            or completion.get('original_files_preserved') is not True
            or checkpoint_receipt.get('checkpoint_sha256') != ORIGINAL2000_SHA):
        raise ValueError('Original2000 source ledger is not backed by a completed preserved run')
    fitted = json.loads((args.b_dir/'completed.json').read_text())
    launch = json.loads((args.b_dir/'launch.json').read_text())
    artifact = torch.load(args.b_dir/'variant-B-native-operators.pt',map_location='cpu',weights_only=True)
    receipt = json.loads((args.b_dir/'variant-B-receipt.json').read_text())
    if (fitted.get('complete') is not True or fitted.get('failure') is not None
            or fitted.get('files_preserved') is not True or not all(fitted.get('states_preserved',{}).values())
            or fitted.get('calibration_sources') != 72 or fitted.get('development_sources') != 96
            or fitted.get('neural_training_updates') != 0
            or launch.get('fit_source_ids') != calibration_ids or launch.get('development_source_ids') != development_ids
            or launch.get('original_step0_sha256') != STEP0_SHA
            or receipt.get('candidate_state_sha256') != B_STATE_SHA
            or receipt.get('full_group_widths') != [384,256,128]):
        raise ValueError('B calibration completion or train/development partition differs')
    saved_b = json.loads((args.b_dir/'variant-B-development.json').read_text())
    if fitted['results']['B']['aggregate'] != saved_b['aggregate']:
        raise ValueError('B saved development result differs from completion')
    references = {step:json.loads(((args.step0.parent if step <= 1000 else args.original2000.parent)/
        f'development-step{step}.json').read_text()) for step in (0,500,1000,1500,2000)}
    # The old journals independently bind exposure order to the original executed updates.
    executed = []
    for folder,begin,end in ((args.step0.parent,1,1000),(args.original2000.parent,1001,2000)):
        records = [json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines() if line.strip()]
        if [r.get('step') for r in records] != list(range(begin,end+1)):
            raise ValueError('Original training journal is incomplete or duplicates an update')
        for row in records:
            ids = row.get('source_ids')
            if ids != list(source_ids[(row['step']-1)*12:row['step']*12]):
                raise ValueError('Original executed source order differs from its checkpoint ledger')
            executed.extend(ids)
    if executed != list(source_ids): raise ValueError('Original exposure was not the complete24,000 prefix')
    protected = {**zero['identity']['protected'],**original['identity']['protected'],**launch['protected']}
    for path,checksum in protected.items():
        if base.sha(path) != checksum: raise ValueError('Preserved input changed: '+path)
    return zero,artifact,receipt,source_ids,recipe,saved_b,references,protected


def main():
    import reconstruction_b_monitor as monitor_module
    import group_model
    import fresh_training_data
    from unified_monitor import UnifiedMonitor
    from joint_recovery_gates_v2 import summarize_regions
    from compare_accumulation_v1 import numeric_reference_check
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('step0','original2000','preserved5000','b-dir','manifest','source-plan','shards','assets','recipe','schedule','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tensorboard',type=Path)
    parser.add_argument('--shard-wait-seconds',type=int,default=3600)
    args = parser.parse_args()
    logdir = (args.tensorboard or args.out/'tensorboard').resolve()
    if args.out.exists() or logdir.exists(): raise FileExistsError('B recovery needs new output and TensorBoard directories')
    retained = (args.step0.parent,args.original2000.parent,args.preserved5000.parent,args.b_dir,args.recipe.parent)
    if any(new.is_relative_to(old.resolve()) or old.resolve().is_relative_to(new)
           for new in (args.out.resolve(),logdir) for old in retained):
        raise ValueError('B output would overlap a preserved run')
    base.policy()
    manifest,pools,_ = base.load_data(args.manifest)
    if len(pools['calibration']) != 72 or len(pools['development']) != 96:
        raise ValueError('Use the original72 calibration and96 development sources')
    fresh = fresh_training_data.FreshTrainingData(args.source_plan,args.manifest,pools,args.shards)
    data = prior.SourceStream(pools,fresh)
    zero,artifact,b_receipt,source_ids,recipe,saved_b,references,protected = authenticate_inputs(args,data,pools)
    modules = (prior,progressive,control,base,screen,replay,joint,group_model,fresh_training_data,monitor_module)
    paths = [args.step0,args.original2000,args.preserved5000,args.original2000.with_suffix('.json'),
        args.step0.with_suffix('.json'),args.preserved5000.with_suffix('.json'),args.manifest,args.source_plan,
        args.recipe,args.schedule,Path(manifest['cache_path']),Path(manifest['overlay_receipt_path']),Path(__file__),
        *[Path(m.__file__) for m in modules],
        *[args.b_dir/name for name in ('launch.json','completed.json','variant-B-native-operators.pt',
                                      'variant-B-receipt.json','variant-B-development.json')]]
    for folder,steps in ((args.step0.parent,(0,500,1000)),(args.original2000.parent,(1500,2000))):
        paths += [folder/'launch.json',folder/'completed.json',folder/'train.jsonl']
        paths += [folder/f'development-step{step}.json' for step in steps]
    protected.update({str(path.resolve()):base.sha(path) for path in paths})
    process = control.gpu_idle_snapshot()
    if str(torch.__version__) != recipe['torch'] or torch.backends.cudnn.version() != recipe['cudnn']:
        raise ValueError('Use the original qualified Torch/cuDNN runtime')
    teacher = base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    model = progressive.initialize_from_teacher(teacher.model.decoder,zero['selection'])
    optimizer,installation = restore_b_start(model,teacher,zero,artifact,b_receipt,[c['source_id'] for c in pools['calibration']])
    if len(base.parameters(model)) != 90 or any(p.requires_grad for p in teacher.model.parameters()):
        raise RuntimeError('Expected all90 native group parameters and a fully frozen teacher')
    teacher_hash = control.state_hash(teacher.model)
    frozen,teacher_frozen = screen.frozen_versions(model),resume.continuation.teacher_versions(teacher)
    common = base.objective(); metadata = {r['source_id']:r for r in manifest['splits']['development']['rows']}
    evaluator = UnifiedMonitor(None,{},metadata)
    boundary_panel,boundary_selection = base.select_boundary_panel(pools['development'],metadata)
    # Reserve all three group checkpoints and modest compact logs, without decoder copies.
    parameter_bytes = sum(v.numel()*v.element_size() for v in model.group_state_dict().values())
    required = parameter_bytes*7+64*1024**2
    ancestor = args.out.parent
    while not ancestor.exists(): ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free < required: raise RuntimeError('Insufficient disk for step0 plus1000/2000 group+Adam checkpoints')
    args.out.mkdir(parents=True)
    monitor = monitor_module.RecoveryMonitor(logdir,source_metadata=metadata)
    screen.restore_rng(zero['rng'])
    identity = {'version':VERSION,'initialization':'Original step0 plus frozen reconstruction B native operators',
        'original_step0_sha256':STEP0_SHA,'b_operators_sha256':B_SHA,'initial_decoder_state_sha256':B_STATE_SHA,
        'original2000_sha256':ORIGINAL2000_SHA,'preserved5000_sha256':STEP5000_SHA,
        'selection':zero['selection'],'schedule_sha256':base.sha(args.schedule),
        'coefficients':dict(prior.COEFFICIENTS),'learning_rate':3e-5,'optimizer_betas':[.9,.99],
        'optimizer_eps':1e-8,'weight_decay':0,'execution_batch_size':1,'gradient_accumulation':12,
        'trainable_stages':[2,3,4],'optimizer_reset_after_start':False,'source_interval':[0,24000],
        'source_ids':list(source_ids),'source_ids_sha256':screen.digest(list(source_ids)),
        'source_plan_identity_sha256':fresh.identity,'original_teacher_forever':True,'protected':protected,
        'rng_origin':'Exact saved original step0 RNG after original setup; restored after B construction and monitoring',
        'original_policy_seed':20260910,'backend':replay.backend_state(),'torch':str(torch.__version__),
        'cudnn':torch.backends.cudnn.version(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'precision':'FP32 TF32 disabled','process_snapshot':process,'installation':installation,
        'review_steps':list(REVIEW_STEPS),'checkpoint_steps':list(CHECKPOINT_STEPS),'target_updates':UPDATES,
        'tensorboard':str(logdir),'disk_free_before':free,'estimated_checkpoints_and_logs_bytes':required,
        'boundary_panel':boundary_selection,'automatic_next_cut':False,'automatic_promotion':False,
        'historical_bitwise_trajectory_claim':False,'no_quality_based_early_stopping':True}
    base.write_json(args.out/'launch.json',identity)
    seen = []; samples = 0; warmed = set(); reports = {}; last_receipt = None
    started = time.monotonic(); update_seconds = validation_seconds = waiting_seconds = 0.
    failure = None; status = 'running'
    def assert_frozen():
        if screen.frozen_versions(model) != frozen or resume.continuation.teacher_versions(teacher) != teacher_frozen:
            raise RuntimeError('A frozen teacher or outer student tensor changed')
    def evaluate(step):
        nonlocal validation_seconds
        before = time.monotonic()
        def capture(rows):
            control._write_rows(args.out/f'quiet-windows-step{step}.jsonl.gz',rows)
            return summarize_regions(rows)
        with replay.diagnostic_state_guard(model,teacher,optimizer):
            report = joint.evaluate_review(evaluator,model,teacher,pools['development'],common,summarize=capture)
            boundary = base.evaluate_boundaries(teacher,model,boundary_panel,zero['selection'],base.batch,base.teacher_forward)
        validation_seconds += time.monotonic()-before
        assert_frozen()
        base.write_json(args.out/f'development-step{step}.json',report)
        base.write_json(args.out/f'boundaries-step{step}.json',boundary)
        return report
    try:
        prior.warm_student(model,teacher,pools['development'],warmed,optimizer)
        initial = evaluate(0)
        comparisons = {'quality':numeric_reference_check(initial,saved_b),
            'rng':replay.compare_tree(screen.rng_state(),zero['rng']),
            'optimizer':replay.compare_tree(optimizer.state_dict(),zero['optimizer']),
            'candidate_state_exact':control.state_hash(model.decoder) == B_STATE_SHA}
        base.write_json(args.out/'initial-parity.json',comparisons)
        if (not comparisons['quality']['passed'] or not comparisons['rng']['equal']
                or not comparisons['optimizer']['equal'] or not comparisons['candidate_state_exact']):
            raise RuntimeError('B state, source/quiet panel, fresh Adam, RNG or saved development baseline did not reproduce')
        reports[0] = initial; monitor.log_validation(initial,0,references[0])
        last_receipt = save_checkpoint(args.out/'checkpoint-step0.pt',model,optimizer,0,seen,identity,zero['selection'],samples)
        last_receipt.update({'frozen_state_preserved':True,'quality':initial['aggregate']})
        base.write_json(args.out/'checkpoint-step0.json',last_receipt)
        for step in range(1,UPDATES+1):
            before = time.monotonic()
            crops = resume.take_when_ready(data,(step-1)*12,12,step=step,monitor=monitor,wait_seconds=args.shard_wait_seconds)
            waiting_seconds += time.monotonic()-before
            ids = [c['source_id'] for c in crops]
            if ids != list(source_ids[(step-1)*12:step*12]): raise RuntimeError('Original ordered crop prefix changed')
            prior.warm_student(model,teacher,crops,warmed,optimizer)
            before = time.monotonic()
            values,checks = perform_update(model,teacher,crops,common,optimizer,diagnostics=step in (1,1001) or step%25 == 0)
            duration = time.monotonic()-before; update_seconds += duration
            seen.extend(ids); samples += sum(c['valid_scored_samples'] for c in crops)
            record = {'step':step,'updates':step,'global_step':step,**values,'source_ids':ids,'unique_sources':len(seen),
                'audio_hours':samples/48000/3600,'step_seconds':duration,'elapsed_seconds':time.monotonic()-started,
                'training_update_seconds':update_seconds,'validation_seconds':validation_seconds,
                'waiting_seconds':waiting_seconds,'teacher_cache_checks':checks}
            with (args.out/'train.jsonl').open('a') as handle: handle.write(json.dumps(record,allow_nan=False)+'\n')
            monitor.log_training(record,step)
            if step == 1 or step%25 == 0:
                base.event('b_recovery_progress',step=step,target=UPDATES,total=values['total'],sources=len(seen),
                    training_update_seconds=update_seconds,elapsed_seconds=record['elapsed_seconds'])
            if step in REVIEW_STEPS:
                report = evaluate(step); reports[step] = report
                monitor.log_validation(report,step,references.get(step))
                data.assert_unchanged()
                if step in CHECKPOINT_STEPS:
                    last_receipt = save_checkpoint(args.out/f'checkpoint-step{step}.pt',model,optimizer,step,seen,identity,zero['selection'],samples)
                    last_receipt.update({'frozen_state_preserved':True,'quality':report['aggregate']})
                    base.write_json(args.out/f'checkpoint-step{step}.json',last_receipt)
                base.write_json(args.out/f'review-step{step}.json',{'step':step,'source_count':len(seen),
                    'aggregate':report['aggregate'],'quiet_regions':report['quiet_regions'],
                    'original_reference_available':step in references,
                    'original_reference_aggregate':references[step]['aggregate'] if step in references else None,
                    'training_update_seconds':update_seconds,'validation_seconds':validation_seconds,
                    'waiting_seconds':waiting_seconds,'elapsed_seconds':time.monotonic()-started,
                    'automatic_next_cut':False})
                base.event('b_recovery_review_ready',step=step,quality=report['aggregate'],automatic_next_cut=False)
        status = 'awaiting_review'
    except BaseException as exc:
        failure = repr(exc); status = 'failed'; raise
    finally:
        monitor.close()
        preserved = screen.frozen_versions(model) == frozen and resume.continuation.teacher_versions(teacher) == teacher_frozen
        preserved = preserved and control.state_hash(teacher.model) == teacher_hash
        data.assert_unchanged()
        files = all(base.sha(path) == checksum for path,checksum in protected.items())
        base.write_json(args.out/'completed.json',{'version':VERSION,'status':status,'failure':failure,
            'step':len(seen)//12,'source_count':len(seen),'scored_samples':samples,
            'last_checkpoint_sha256':last_receipt['checkpoint_sha256'] if last_receipt else None,
            'training_update_seconds':update_seconds,'validation_seconds':validation_seconds,'waiting_seconds':waiting_seconds,
            'elapsed_seconds':time.monotonic()-started,'frozen_state_preserved':preserved,'original_files_preserved':files,
            'review_steps_completed':list(reports),'optimizer_reset_after_start':False,'automatic_next_cut':False,
            'automatic_promotion':False,'final':reports[max(reports)]['aggregate'] if reports else None})
        if not preserved or not files: raise RuntimeError('B recovery changed a protected original asset or frozen tensor')


if __name__ == '__main__': main()
