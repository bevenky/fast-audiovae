"""Fresh, matched optimizer trials with the existing fixed startup protection.

Qualification ranks a separate training-calibration probe, never development.
Every process reconstructs the sealed teacher-derived initializer and empty
optimizer. Existing sources, checkpoints and numerical policies are read-only.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import time

import torch
import grail_startup_recovery as original
import recovery_optimizers as optimizers
import startup_mixed_optimizer_update as mixed

base, screen, replay, prior = original.base, original.screen, original.replay, original.prior
control, joint, startup = original.control, original.joint, original.startup
VERSION = 'audiovae2_optimizer_comparison_v1'
REVIEWS = (0, 250, 500, 1000, 1500, 2000)
FULL_CHECKPOINTS = (0, 1000, 2000)
GROUP_CHECKPOINTS = (500, 1500)


def write(path, value):
    base.write_json(path, value)


def numeric_parity(old, new):
    """All original non-timing scalar fields must exactly reproduce."""
    ignored = {'step_seconds', 'elapsed_seconds', 'training_update_seconds',
               'validation_seconds', 'waiting_seconds', 'q_step_seconds',
               'q_ordinary_update_seconds', 'q_auxiliary_seconds'}
    keys = [k for k in old if k not in ignored]
    failed = [k for k in keys if k not in new or old[k] != new[k]]
    return {'passed': not failed, 'compared': len(keys), 'mismatched_fields': failed}


def validate_recovery_qualification(trial, config_path):
    """A full arm must be bound to the completed shared qualification record."""
    path=Path(trial['qualification_path']); decision=original.read(path)
    if (base.sha(path)!=trial.get('qualification_sha256')
            or decision.get('version')!='audiovae2_optimizer_qualification_v1'
            or decision.get('all_candidates_attempted') is not True
            or decision.get('control_parity_passed') is not True
            or decision.get('development_used_for_selection') is not False
            or decision.get('config_sha256')!=base.sha(Path(config_path))
            or decision.get('source_manifest_sha256')!=base.sha(Path(trial['source_manifest']))):
        raise ValueError('Recovery requires a sealed complete qualification decision')
    selected=decision['selected'][trial['method']]
    if (selected['optimizer_kwargs']!=trial['optimizer_kwargs']
            or selected.get('eligible') is not True
            or base.sha(Path(selected['completed_path']))!=selected['completed_sha256']):
        raise ValueError('Recovery configuration differs from its qualified candidate')
    return path


def save_checkpoint(path, model, optimizer, step, identity, seen, samples, *, group_only):
    if path.exists() or path.with_suffix('.tmp').exists():
        raise FileExistsError('Never replace a prior snapshot')
    params = base.parameters(model)
    optimizers.assert_optimizer_state(optimizer, params, step)
    if seen != identity['source_ids'][:step*12] or len(seen) != step*12:
        raise RuntimeError('Snapshot source prefix changed')
    group = {k:v.detach().cpu().clone() for k,v in model.group_state_dict().items()}
    if len(group) != 90 or any(not torch.isfinite(v).all() for v in group.values()):
        raise RuntimeError('Snapshot native tensors are invalid')
    # Conservative bound includes matrix factors and serialization temporaries.
    minimum = sum(v.numel()*v.element_size() for v in group.values()) * (1 if group_only else 5) + 48*1024**2
    if shutil.disk_usage(path.parent).free < minimum:
        raise RuntimeError('Insufficient free space for preserved snapshot')
    payload = {'format':VERSION, 'step':step, 'group':group,
        'optimizer_config':optimizers.optimizer_config(optimizer), 'identity':identity,
        'sources_seen':list(seen), 'scored_samples':samples, 'selection':model.selections,
        'snapshot_kind':'group_only' if group_only else 'full_optimizer',
        'automatic_next_cut':False, 'automatic_promotion':False}
    if not group_only:
        payload.update(optimizer=optimizer.state_dict(), rng=screen.rng_state())
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp); tmp.replace(path)
    return {'step':step, 'sha256':base.sha(path), 'bytes':path.stat().st_size,
            'group_tensors':90, 'source_count':len(seen), 'snapshot_kind':payload['snapshot_kind']}


class Monitor:
    """One run per trial, with identical tags for comparing optimizer curves."""
    def __init__(self, path, label, target):
        from torch.utils.tensorboard import SummaryWriter
        if path.exists():
            raise FileExistsError('Use fresh TensorBoard events for every trial')
        self.writer = SummaryWriter(str(path), flush_secs=10)
        self.target = target
        self.writer.add_text('Guide', label + '. Fresh sealed initialization; unchanged 12 startup constraints. '
            'Learning-rate selection uses a training-calibration probe. Development is evaluation-only. '
            'Quiet cohorts overlap. Correlation is not perceptual accuracy. Each curve is one independent run.', 0)
        self.writer.add_custom_scalars({'Optimizer comparison':{
            'Waveform correlation':['Multiline',['quality/nonquiet_cosine_mean']],
            'Quiet passing':['Multiline',['quality/quiet_passing_percent']],
            'Startup passing':['Multiline',['quality/calibration_startup_percent','quality/development_startup_percent']],
            'Training objective':['Multiline',['loss/total']]}})

    def training(self, row):
        step = row['step']
        for key in ('total','waveform','mel','feature'):
            self.writer.add_scalar('loss/'+key, row[key], step)
        for key in ('q_actual_norm_over_optimizer','q_accepted_displacement_norm',
                    'q_normal_accepted','q_zero_displacement','q_auxiliary_seconds',
                    'q_ordinary_update_seconds','step_seconds'):
            self.writer.add_scalar('update/'+key, row[key], step)
        self.writer.add_scalar('progress/percent', 100*step/self.target, step)

    def quality(self, report, step):
        for key, value in report['aggregate'].items():
            if isinstance(value, (int,float)) and value is not None:
                self.writer.add_scalar('quality/'+key, value, step)
        for region, row in report['regions'].items():
            for key in ('windows','failed','residual_rms','output_rms','teacher_rms'):
                if row.get(key) is not None:
                    self.writer.add_scalar('quiet/'+region+'/'+key, row[key], step)
            self.writer.add_scalar('quiet/'+region+'/passing_percent',100*(1-row['failed']/row['windows']),step)
        all_quiet = report['regions']['all_quiet']
        self.writer.add_scalar('quality/quiet_passing_percent',100*(1-all_quiet['failed']/all_quiet['windows']),step)
        for key in ('calibration_startup','development_startup'):
            self.writer.add_scalar('quality/'+key+'_percent',100*report[key]['passed']/report[key]['windows'],step)
        for key in ('active_rms_ratio','whistle_rms_ratio'):
            if report.get(key) is not None:self.writer.add_scalar('quality/'+key,report[key],step)
        self.writer.flush()

    def close(self):
        self.writer.flush(); self.writer.close()


def main():
    from unified_monitor import UnifiedMonitor
    from joint_recovery_gates_v2 import summarize_regions
    from compare_accumulation_v1 import numeric_reference_check
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tensorboard',type=Path,required=True)
    args = parser.parse_args()
    config, trial = original.read(args.config), original.read(args.trial)
    target = trial['updates']; method = trial['method']; qualification = target==64
    if target not in (64,2000) or method not in optimizers.METHODS:
        raise ValueError('Only matched64 qualification or bounded2000 recovery is supported')
    qualification_path=None if qualification else validate_recovery_qualification(trial, args.config)
    if args.out.exists() or args.tensorboard.exists():
        raise FileExistsError('Every trial must use fresh output and events')
    retained=[Path(config[k]).resolve() for k in ('initializer_dir','assets','shards')]
    retained += [Path(config[k]).resolve().parent for k in ('original_step0','original2000','recipe')]
    if any(new.is_relative_to(old) or old.is_relative_to(new)
           for new in (args.out.resolve(),args.tensorboard.resolve()) for old in retained):
        raise ValueError('Comparison outputs overlap preserved inputs')
    if shutil.disk_usage(args.out.parent).free < (96 if qualification else 370)*1024**2:
        raise RuntimeError('Insufficient retained-artifact headroom')
    source_manifest = original.read(Path(trial['source_manifest']))
    required={'optimizer_comparison_runner.py','startup_mixed_optimizer_update.py','recovery_optimizers.py'}
    if not required.issubset(source_manifest) or any(Path(n).name!=n or len(h)!=64
            or any(c not in '0123456789abcdef' for c in h) for n,h in source_manifest.items()):
        raise ValueError('Comparison source manifest lacks its exact executable scope')
    for name, expected in source_manifest.items():
        if base.sha(Path(__file__).parent/name)!=expected:
            raise RuntimeError('Comparison source differs from its sealed manifest')
    # Reuse the authenticated original factory, data and exact initial Adam/RNG.
    built = original.build(config)
    manifest,pools,data,zero,teacher,model,old_optimizer,source_ids,reference,protected,installation,process = built
    optimizer = optimizers.build_optimizer(model, method, **trial['optimizer_kwargs'])
    optimizers.assert_optimizer_state(optimizer,base.parameters(model),0)
    if method=='adamw' and trial['optimizer_kwargs']['matrix_lr']==3e-5:
        if not replay.compare_tree(optimizer.state_dict(),old_optimizer.state_dict())['equal']:
            raise RuntimeError('Exact AdamW control differs before its first update')
    del old_optimizer
    screen.restore_rng(zero['rng'])
    initial_hash = control.state_hash(model.decoder)
    if initial_hash!=config['initializer_state_sha256']:
        raise RuntimeError('Fresh optimizer construction changed the sealed initializer')
    original_modules = (original, original.init, original.b, original.corrected,
        original.corrected.complete, original.corrected.complete.projection,
        original.corrected.correction, original.corrected.correction.signed,
        original.corrected.anchor, original.corrected.q, original.corrected.anchor.warmup,
        original.corrected.grid, prior,base,screen,replay,joint,control,startup)
    for module in original_modules:
        path = Path(module.__file__)
        if path.name in original.POLICY_PINS and base.sha(path)!=original.POLICY_PINS[path.name]:
            raise RuntimeError('A frozen original policy source changed')
        protected[str(path.resolve())] = base.sha(path)
    for path in (args.config,args.trial,Path(trial['source_manifest'])):
        protected[str(path.resolve())]=base.sha(path)
    if qualification_path is not None:
        protected[str(qualification_path.resolve())]=base.sha(qualification_path)
        selected_report=Path(original.read(qualification_path)['selected'][method]['completed_path'])
        protected[str(selected_report.resolve())]=base.sha(selected_report)
    for name, expected in source_manifest.items():
        protected[str((Path(__file__).parent/name).resolve())]=expected
    teacher_hash=control.state_hash(teacher.model)
    frozen=screen.frozen_versions(model); teacher_versions=original.b.resume.continuation.teacher_versions(teacher)
    args.out.mkdir()
    identity = {'version':VERSION, 'trial':trial, 'optimizer':optimizers.optimizer_config(optimizer),
        'initial_state_sha256':initial_hash, 'initial_rng_sha256':original.rng_digest(zero['rng']),
        'initializer_artifact_sha256':config['initializer_pins'][original.INIT_FILES[0]],
        'source_ids':list(source_ids[:target*12]), 'source_ids_sha256':screen.digest(list(source_ids[:target*12])),
        'source_plan_identity_sha256':data.fresh.identity, 'selection':model.selections,
        'teacher_source_sha256':base.SOURCE_SHA256,'teacher_checkpoint_sha256':base.CHECKPOINT_SHA256,
        'coefficients':dict(prior.COEFFICIENTS),'execution_batch_size':1,'accumulation':12,
        'recurring_calibration_sources':6, 'probe_sources':12,
        'learning_rate_selection':'Fixed training-calibration probe; development excluded',
        'pilot_state_resumed':False,'automatic_next_cut':False,'automatic_promotion':False,
        'installation':installation,'protected':protected,'torch':str(torch.__version__),
        'cudnn':torch.backends.cudnn.version(),'backend':replay.backend_state(), 'process_snapshot':process}
    metadata = {r['source_id']:r for r in manifest['splits']['development']['rows']}
    monitor = Monitor(args.tensorboard,trial['name'],target)
    report = {'version':VERSION,'complete':False,'status':'running','updates':0,
        'trial_name':trial['name'],'method':method,'optimizer':identity['optimizer'],
        'initial_state_sha256':initial_hash,'initial_rng_sha256':identity['initial_rng_sha256'],
        'initializer_artifact_sha256':identity['initializer_artifact_sha256'],
        'source_plan_identity_sha256':data.fresh.identity,'pilot_state_resumed':False,
        'automatic_promotion':False,'neural_training_target':target,'failure_category':None,
        'milestone_quality':{},'checkpoint_receipts':{},'parity':None}
    seen=[];samples=0;warmed=set();started=time.monotonic();update_seconds=0.;validation_seconds=0.
    old_fractions=(mixed.q.FRACTIONS,mixed.anchor.FRACTIONS)
    old_backtrack,old_score=mixed.q.backtrack,mixed.q.score_entries
    reference64 = original.read(Path(trial['parity_reference'])) if trial.get('parity_reference') else None
    if reference64:
        reference_path=Path(trial['parity_reference'])
        if (not qualification or method!='adamw' or trial['optimizer_kwargs']['matrix_lr']!=3e-5
                or base.sha(reference_path)!=trial.get('parity_reference_sha256')
                or reference64.get('version')!=original.PILOT_VERSION
                or reference64.get('method')!='corrected' or reference64.get('complete') is not True
                or reference64.get('updates')!=64 or len(reference64.get('update_records',[]))!=64
                or reference64.get('initial_state_sha256')!=initial_hash
                or reference64.get('initial_rng_sha256')!=identity['initial_rng_sha256']
                or reference64.get('ordinary_source_prefix_sha256')!=identity['source_ids_sha256']
                or reference64.get('all_preservation_checks_passed') is not True):
            raise ValueError('Exact control replay requires the authenticated corrected64 reference')
        protected[str(reference_path.resolve())]=base.sha(reference_path)
    write(args.out/'launch.json',identity)
    def assert_frozen():
        if screen.frozen_versions(model)!=frozen or original.b.resume.continuation.teacher_versions(teacher)!=teacher_versions:
            raise RuntimeError('Frozen model tensors changed')
    def evaluate(step):
        nonlocal validation_seconds
        tick=time.monotonic()
        with replay.diagnostic_state_guard(model,teacher,optimizer):
            full=joint.evaluate_review(UnifiedMonitor(None,{},metadata),model,teacher,pools['development'],common,summarize=summarize_regions)
        summary=original.init.summarize_quality(full)
        summary['calibration_startup']=startup.score_panel(model,anchors)
        summary['development_startup']=startup.score_panel(model,development)
        selected=[r for sid,r in full['overview_window_metrics']['by_source'].items()
            if 'human_whistling_source_description' in metadata.get(sid,{}).get('verified_source_labels',[])]
        te=sum(r['active_teacher_energy'] for r in selected);se=sum(r['active_student_energy'] for r in selected)
        summary['whistle_rms_ratio']=math.sqrt(se/te) if te else None
        validation_seconds+=time.monotonic()-tick;assert_frozen()
        write(args.out/f'development-step{step}.json',full)
        write(args.out/f'review-step{step}.json',summary)
        report['milestone_quality'][str(step)]=summary;monitor.quality(summary,step)
        return summary
    try:
        with mixed.grid.extended_grid():
            anchors,anchor_cache=startup.startup_panel(teacher,pools['calibration'],6)
            development,development_cache=startup.startup_panel(teacher,pools['development'],13)
            updater=mixed.StartupAnchorUpdate(model,teacher,anchors,optimizer)
            anchor_ids={e['crop']['source_id'] for e in anchors}
            probe=[c for c in pools['calibration'] if c['source_id'] not in anchor_ids][:12]
            if len(probe)!=12 or set(c['source_id'] for c in probe)&set(source_ids[:target*12]):
                raise RuntimeError('Training probe must be disjoint from ordinary stream and six anchors')
            common=base.objective();report['initialization']=updater.receipt
            report['startup_cache_checks']={'calibration':anchor_cache,'development':development_cache}
            prior.warm_student(model,teacher,pools['development'],warmed,optimizer)
            initial=evaluate(0)
            quality_keys=('aggregate','regions','active_rms_ratio','teacher_window_identity_sha256','calibration_startup')
            baseline={k:initial[k] for k in quality_keys}
            if not numeric_reference_check(baseline,{k:reference[k] for k in quality_keys})['passed']:
                raise RuntimeError('Fixed initial quality failed to reproduce')
            prior.warm_student(model,teacher,probe,warmed,optimizer)
            with replay.diagnostic_state_guard(model,teacher,optimizer),torch.no_grad():
                report['training_probe_before']=startup.ordinary_objective(model,teacher,probe,common)
            report['training_probe_identity_sha256']=screen.digest([c['source_id'] for c in probe])
            first=data.take(0,12);prior.warm_student(model,teacher,first,warmed,optimizer)
            screen.restore_rng(zero['rng'])
            if not qualification:
                receipt=save_checkpoint(args.out/'checkpoint-step0.pt',model,optimizer,0,identity,[],0,group_only=False)
                write(args.out/'checkpoint-step0.json',receipt);report['checkpoint_receipts']['0']=receipt
            for step in range(1,target+1):
                crops=data.take((step-1)*12,12);ids=[c['source_id'] for c in crops]
                if ids!=list(source_ids[(step-1)*12:step*12]) or set(ids)&set(seen):
                    raise RuntimeError('Ordinary source order changed or repeated')
                prior.warm_student(model,teacher,crops,warmed,optimizer)
                tick=time.monotonic();values,checks=updater.perform_update(model,teacher,crops,common,optimizer,diagnostics=True)
                duration=time.monotonic()-tick;update_seconds+=duration
                if len(checks)!=12 or not all(c['allclose_original_tolerance'] for c in checks):
                    raise RuntimeError('Original teacher target cache changed')
                assert_frozen();seen.extend(ids);samples+=sum(c['samples'] for c in checks)
                row={k:v for k,v in values.items() if k.startswith(('q_','startup_anchor_')) or k in (*prior.COEFFICIENTS,'total')}
                row.update(step=step,step_seconds=duration,elapsed_seconds=time.monotonic()-started,
                    unique_sources=len(seen),audio_hours=samples/48000/3600,training_update_seconds=update_seconds,
                    validation_seconds=validation_seconds,waiting_seconds=0.)
                if reference64:
                    parity=numeric_parity(reference64['update_records'][step-1],row)
                    if not parity['passed']:report['parity']=parity;raise RuntimeError('Adam control differs from original non-timing trace')
                with (args.out/'train.jsonl').open('a') as handle:
                    handle.write(json.dumps({**row,'source_ids':ids,'teacher_cache_checks':checks},allow_nan=False)+'\n')
                report['updates']=step;monitor.training(row)
                if step%25==0:base.event('optimizer_comparison_progress',trial=trial['name'],step=step,target=target)
                if (qualification and step==target) or (not qualification and step in REVIEWS):
                    evaluate(step)
                if not qualification and step in (*FULL_CHECKPOINTS,*GROUP_CHECKPOINTS):
                    group_only=step in GROUP_CHECKPOINTS
                    receipt=save_checkpoint(args.out/f'checkpoint-step{step}.pt',model,optimizer,step,identity,seen,samples,group_only=group_only)
                    write(args.out/f'checkpoint-step{step}.json',receipt);report['checkpoint_receipts'][str(step)]=receipt
            with replay.diagnostic_state_guard(model,teacher,optimizer),torch.no_grad():
                report['training_probe_after']=startup.ordinary_objective(model,teacher,probe,common)
            optimizers.assert_optimizer_state(optimizer,base.parameters(model),target)
            matrix_state=[optimizer.state[p] for name,p in model.group_named_parameters()
                          if name in optimizers.MATRIX_NAMES]
            report['optimizer_activity']={'matrix_parameters':len(matrix_state),
                'matrix_steps':[int(s['step']) for s in matrix_state],
                'shampoo_last_refresh_steps':[int(s['last_refresh']) for s in matrix_state if 'last_refresh' in s],
                'nor_neuron_state_shapes':[list(s['variance_neuron'].shape) for s in matrix_state if 'variance_neuron' in s]}
            report['parity']={'passed':True,'updates':target} if reference64 else None
            report['complete']=True;report['status']='awaiting_comparison'
    except BaseException as exc:
        report['failure_category']=type(exc).__name__;report['status']='failed';raise
    finally:
        monitor.close();data.assert_unchanged()
        preserved={'teacher':control.state_hash(teacher.model)==teacher_hash,
            'frozen_outer':screen.frozen_versions(model)==frozen,
            'files':all(base.sha(p)==h for p,h in protected.items()),
            'fraction_globals':(mixed.q.FRACTIONS,mixed.anchor.FRACTIONS)==old_fractions,
            'score_globals':mixed.q.backtrack is old_backtrack and mixed.q.score_entries is old_score}
        report.update(preserved=preserved,all_preservation_checks_passed=all(preserved.values()),
            ordinary_unique_sources=len(seen),ordinary_source_prefix_sha256=screen.digest(seen),
            scored_samples=samples,recurring_calibration_sources=6,anchor_update_participations=6*len(seen)//12,
            elapsed_seconds=time.monotonic()-started,training_update_seconds=update_seconds,
            validation_seconds=validation_seconds,free_bytes_after=shutil.disk_usage(args.out).free)
        report['complete']=bool(report['complete'] and all(preserved.values()) and len(seen)==target*12)
        write(args.out/'completed.json',report)
        if not all(preserved.values()):raise RuntimeError('Immutable experiment preservation check failed')


if __name__=='__main__':main()
