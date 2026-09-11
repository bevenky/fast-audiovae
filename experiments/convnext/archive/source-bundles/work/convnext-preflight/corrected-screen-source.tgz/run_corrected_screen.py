"""Four matched debugging forks, preserving the original step8090 checkpoint."""
from __future__ import annotations
import copy, gc, json, os, shutil, time, types
from dataclasses import asdict, replace, fields
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from audiovae_student.cache import TrainingCrop, sample_training_crop
from audiovae_student.data import ManifestRow, load_manifest
from audiovae_student.prepare_targets import _target_teacher
from audiovae_student.teacher import FrozenAudioVAE2
from audiovae_student.source_corpus import SourceCorpus
from audiovae_student.representative_pilot import _record
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import file_sha, digest, PilotWindow
from audiovae_student.training import _restore_rng, _rng_state
from audiovae_student.fusion_migration import build_fusion_engine
from audiovae_student.fusion_evaluation import evaluate_fusion, diagnose_streaming
from audiovae_student.corrected_calibration import calibrate_changed_losses, audit_output_gradients
from audiovae_student.recipe_v2 import scored_batch_v2
from audiovae_student.discriminators import discriminator_loss
from corrected_component_diagnostics import diagnose_components
import run_fusion_screen as previous

BASE=Path('/workspace/fast-audiovae-convnext-20260909-r9')
OLD=Path('/workspace/fast-audiovae-convnext-20260908-r1')
PRIOR=BASE/'remediation/fusion-screen'
OUT=BASE/'remediation/corrected-screen'
STEPS=400;BATCH=32;D_WARMUP=64;CALIBRATION=32
ARMS=(('regular','control','regular_generator',False),
      ('targeted','control','targeted_generator',False),
      ('targeted_magnitude','fresh_magnitude','targeted_generator',True),
      ('targeted_complex','complex','targeted_generator',True))


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temp,path)


def status(**kw):
    kw['time_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    atomic_json(OUT/'status.json',kw);print(json.dumps(kw),flush=True)


def save(path,value):
    if shutil.disk_usage(OUT).free<900*1024**2:
        raise RuntimeError('Insufficient reserve for atomic checkpoint; retain original and prior arms')
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(previous.cpu_tree(value),f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def _key(entry):
    w=entry['window']
    return (w['source_id'],w['start_frame'],w['valid_output_samples48k'])


def planned_views(self,batch):
    predictions=[];targets=[];selected=[]
    for p,t,crop in zip(batch.predictions,batch.targets,batch.crops,strict=True):
        needed=self.config.adversarial_samples
        if p.shape[-1]<needed:raise ValueError('A planned discriminator view is too short')
        random_start=int(torch.randint(p.shape[-1]-needed+1,(),generator=self.crop_generator).item())
        key=(crop.source_id,crop.start_frame,crop.valid_scored_samples)
        entry=self.screen_view_entries.get(key)
        start=entry.get('discriminator_start_sample48k') if entry else None
        if start is None:start=random_start
        if type(start) is not int or not 0<=start<=p.shape[-1]-needed:
            raise ValueError('Planned discriminator interval exceeds valid scored audio')
        predictions.append(p[...,start:start+needed]);targets.append(t[...,start:start+needed])
        selected.append({'source_id':crop.source_id,'start_frame':crop.start_frame,
                         'offset48k':start,'planned':bool(entry and entry.get('discriminator_start_sample48k') is not None)})
    self.screen_last_views=selected
    return torch.cat(predictions),torch.cat(targets),len(predictions)


def bind_views(engine,entries):
    engine.screen_view_entries={_key(e):e for e in entries}
    engine.screen_last_views=[]
    engine._perceptual_audio=types.MethodType(planned_views,engine)


def prepare(parent,data):
    rows={sid:ManifestRow.from_dict(r) for sid,r in data['rows'].items()}
    counts=data['counts']
    pools=data['pools']
    required={'regular_generator':STEPS*BATCH,'targeted_generator':STEPS*BATCH,
              'discriminator_warmup':D_WARMUP*BATCH,'gradient_calibration':CALIBRATION*BATCH}
    for name,n in required.items():
        if len(pools[name])!=n:raise ValueError('Wrong pool length: '+name)
    all_entries={_key(e):e for pool in pools.values() for e in pool}
    teacher=FrozenAudioVAE2.from_files(OLD/'assets/audio_vae_v2.py',OLD/'assets/audiovae.pth',device='cuda')
    first=load_manifest('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1/train-manifest.jsonl')[0]
    teacher=_target_teacher(teacher,first)
    teacher_sha=state_fingerprint(teacher.model.state_dict())
    if teacher_sha!=parent['identity']['data']['teacher_state_sha256']:
        raise ValueError('Teacher changed from parent')
    previous.OUT=OUT
    heldout,metadata,panel_identity=previous.panels(parent,teacher)
    fixture=next(c for c in heldout if c.source_id=='encoded_zero')
    with torch.no_grad():
        repeated1=teacher.decode(fixture.latents.cuda())
        repeated2=teacher.decode(fixture.latents.cuda())
        repeatability={'fixture':'encoded_zero','same_latents_and_full_history':True,
            'repeated_forward_max_abs_error':float((repeated1-repeated2).abs().max()),
            'versus_generated_target_max_abs_error':float((repeated1.cpu()-fixture.teacher_audio).abs().max())}
    del repeated1,repeated2
    heldout_ids={c.source_id for c in heldout}
    if heldout_ids & {key[0] for key in all_entries}:raise ValueError('Heldout source enters training')
    wanted={}
    for key,entry in all_entries.items():wanted.setdefault(key[0],[]).append((key,entry))
    corpus=SourceCorpus(tuple(rows.values()),teacher,cache_dir=OUT/'teacher-cache',
        input_sample_counts=counts,max_disk_bytes=64*1024**2,min_free_bytes=900*1024**2,
        max_memory_utterances=16,allow_prepared_source=True)
    cache={}
    try:
        ids=list(wanted)
        for offset in range(0,len(ids),8):
            group=ids[offset:offset+8]
            corpus.prefetch(group,max_batch_size=8,max_total_input_samples=1_920_000)
            for sid in group:
                record=_record(corpus,rows[sid],counts[sid])
                for key,entry in wanted[sid]:
                    w=PilotWindow(**{f.name:entry['window'][f.name] for f in fields(PilotWindow)})
                    crop=sample_training_crop(record,w.start_frame,w.scored_frames,context_frames=30)
                    cache[key]=replace(crop,valid_scored_samples=w.valid_output_samples48k)
            if offset%240==0:status(state='preparing_teacher_targets',sources=min(offset+8,len(ids)),total_sources=len(ids))
    finally:corpus.close()
    if state_fingerprint(teacher.model.state_dict())!=teacher_sha:raise ValueError('Teacher mutated')
    identity={'teacher_state_sha256':teacher_sha,'teacher':teacher.provenance,'teacher_repeatability':repeatability,
        'target_crops':{name:_crop_identity([cache[_key(e)] for e in pool]) for name,pool in pools.items()},
        'heldout':panel_identity,'all_pool_target_count':len(cache)}
    del teacher;gc.collect();torch.cuda.empty_cache()
    return {name:[cache[_key(e)] for e in pool] for name,pool in pools.items()},heldout,metadata,identity


def warmup(engine,crops,directory):
    fingerprint=state_fingerprint(engine.model.state_dict())
    engine.crop_generator.manual_seed(31904)
    for i in range(D_WARMUP):
        selected=crops[i*BATCH:(i+1)*BATCH]
        engine.model.eval()
        with torch.no_grad():
            batch,_=scored_batch_v2(engine.model,selected,engine.reconstruction,engine.device)
            p,t,n=engine._perceptual_audio(batch)
        if n!=BATCH:raise ValueError('D warmup dropped examples')
        engine.discriminators.train();engine.discriminator_optimizer.zero_grad(set_to_none=True)
        loss=discriminator_loss(engine.discriminators,p,t,
            example_weights=p.new_tensor([c.valid_scored_samples for c in selected]))
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(engine.discriminators.parameters(),engine.config.gradient_clip,error_if_nonfinite=True)
        engine.discriminator_optimizer.step();engine.discriminator_optimizer.zero_grad(set_to_none=True)
        with (directory/'warmup.jsonl').open('a') as f:
            f.write(json.dumps({'update':i+1,'loss':float(loss.detach()),'gradient_norm':float(norm)})+'\n')
        if (i+1)%16==0:status(state='discriminator_warmup',arm=directory.name,updates=i+1,total=D_WARMUP)
    if state_fingerprint(engine.model.state_dict())!=fingerprint:raise ValueError('Warmup changed student')


def snapshot(engine,identity,receipt,name,migration,done):
    return {'format_version':'corrected_screen_v1','engine':engine.state_dict(),'rng':_rng_state(),
        'arm':name,'generator_updates':done,'migration':migration,
        'experiment_identity_sha256':digest(identity),'parent_sha256':receipt['checkpoint_sha256'],
        'data_plan_sha256':identity['data_plan_sha256'],
        'discriminator_only_updates':D_WARMUP if name.startswith('targeted_') else 0}


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    if (OUT/'identity.json').exists():raise RuntimeError('Refusing automatic replay of a started experiment')
    receipt=json.loads((PRIOR/'parent-receipt.json').read_text())
    if file_sha(PRIOR/'parent.pt')!=receipt['checkpoint_sha256']:raise ValueError('Original checkpoint changed')
    parent=torch.load(PRIOR/'parent.pt',map_location='cpu',weights_only=True)
    if parent['engine']['step']!=8090:raise ValueError('Wrong starting step')
    data=json.loads((OUT/'targeted-data.json').read_text())
    pools,heldout,metadata,target_identity=prepare(parent,data)
    identity={'format_version':1,'parent_sha256':receipt['checkpoint_sha256'],'parent_step':8090,
        'data_plan_sha256':file_sha(OUT/'targeted-data.json'),'generator_updates_per_arm':STEPS,
        'batch_size':BATCH,'fresh_discriminator_warmup_updates':D_WARMUP,'fixed_weight_calibration_batches':CALIBRATION,
        'arms':[list(a) for a in ARMS],'targets':target_identity,
        'sources':{str(p.relative_to(Path(__file__).parent)):file_sha(p)
                   for p in Path(__file__).parent.rglob('*.py')},
        'new_student_parameters':0,'student_normalization':'preserve calibrated fixed statistics',
        'replay_policy':'debugging may reuse earlier-run audio; unique scored intervals per arm',
        'inference_architecture_changes':False}
    atomic_json(OUT/'identity.json',identity)
    diagnostic_crops=[c for c in heldout if c.source_id.startswith('encoded_')]
    diagnostic_crops += [c for c in heldout if c.source_id not in {x.source_id for x in diagnostic_crops}
        and (metadata.get(c.source_id,{}).get('group')=='expressive'
             or metadata.get(c.source_id,{}).get('condition') in {'whispering','laughter','screaming','whistling','breathing'})][:30]
    for name,variant,pool_name,calibrate in ARMS:
        directory=OUT/name
        directory.mkdir(exist_ok=False)
        _restore_rng(parent['rng']);torch.manual_seed(93508)
        engine,migration=build_fusion_engine(parent['engine'],variant,device='cuda')
        entries=data['pools'][pool_name]
        if name!='regular':
            entries=entries+data['pools']['discriminator_warmup']+data['pools']['gradient_calibration']
        else:
            # Ordinary random views for all regular training and fixed audits.
            entries=[]
        bind_views(engine,entries)
        atomic_json(directory/'migration.json',migration)
        _restore_rng(parent['rng'])
        status(state='evaluation_before',arm=name)
        before=evaluate_fusion(engine,heldout,metadata)
        atomic_json(directory/'before.json',before)
        atomic_json(directory/'components-before.json',diagnose_components(engine,diagnostic_crops,metadata))
        audit_crops=pools['gradient_calibration'][:BATCH]
        if calibrate:
            warmup(engine,pools['discriminator_warmup'],directory)
            status(state='fixed_weight_gradient_calibration',arm=name)
            report=calibrate_changed_losses(engine,
                [pools['gradient_calibration'][i*BATCH:(i+1)*BATCH] for i in range(CALIBRATION)],
                provenance={'split':'train','data_plan_sha256':identity['data_plan_sha256'],
                    'pool':'gradient_calibration','fresh_discriminator_warmup_updates':D_WARMUP},seed=1307)
            atomic_json(directory/'calibration.json',report)
        atomic_json(directory/'gradients-before.json',audit_output_gradients(engine,audit_crops,seed=1773))
        _restore_rng(parent['rng']);engine.crop_generator.set_state(parent['engine']['crop_rng'].cpu())
        start=time.perf_counter()
        for i in range(STEPS):
            batch=pools[pool_name][i*BATCH:(i+1)*BATCH]
            atomic_json(directory/'inflight.json',{'update':i+1,'crops':[(c.source_id,c.start_frame) for c in batch]})
            metrics=engine.train_step(batch)
            with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
            with (directory/'views.jsonl').open('a') as f:
                f.write(json.dumps({'update':i+1,'views':engine.screen_last_views})+'\n')
            (directory/'inflight.json').unlink()
            if (i+1)%100==0:save(directory/'last.pt',snapshot(engine,identity,receipt,name,migration,i+1))
            if (i+1)%25==0:
                status(state='training',arm=name,updates=i+1,total=STEPS,seconds=time.perf_counter()-start,
                       waveform=metrics['teacher_waveform'],mel=metrics['teacher_mel'])
        elapsed=time.perf_counter()-start
        status(state='evaluation_after',arm=name)
        after=evaluate_fusion(engine,heldout,metadata)
        atomic_json(directory/'after.json',after)
        atomic_json(directory/'components-after.json',diagnose_components(engine,diagnostic_crops,metadata))
        atomic_json(directory/'gradients-after.json',audit_output_gradients(engine,audit_crops,seed=1773))
        # Correctness only, no RTF benchmark. Test both latent chunk durations.
        stream={}
        cpu_model=copy.deepcopy(engine.model).cpu()
        stream_crops=[next(c for c in heldout if c.source_id=='encoded_zero'),
                      next(c for c in heldout if c.source_id!='encoded_zero' and c.latents.shape[-1]>=40)]
        for j,crop in enumerate(stream_crops):
            for frames in (2,4):
                result=diagnose_streaming(cpu_model,crop.latents.cpu(),frames_per_chunk=frames)
                result['numerical_tolerance']=2e-6
                result['passed']=result['sample_count_passed'] and result['max_abs_error']<=2e-6
                if not result['passed']:raise RuntimeError('Trained CPU streaming parity failed')
                stream[f'{j}_{frames}']=result
        del cpu_model
        atomic_json(directory/'streaming.json',stream)
        os.replace(directory/'last.pt',directory/'final.pt')
        atomic_json(directory/'complete.json',{'arm':name,'updates':STEPS,'step':engine.step,'training_seconds':elapsed,
            'checkpoint_sha256':file_sha(directory/'final.pt'),
            'before_summary':before['summary'],'after_summary':after['summary']})
        del engine;gc.collect();torch.cuda.empty_cache()
    if file_sha(PRIOR/'parent.pt')!=receipt['checkpoint_sha256']:raise ValueError('Original checkpoint mutated')
    status(state='complete',arms=[a[0] for a in ARMS],original_run_still_paused=True)


if __name__=='__main__':
    import fcntl
    OUT.mkdir(exist_ok=True)
    try:
        with (BASE/'training-runs/.decoder-recipe-v2-expressive.runner.lock').open('rb') as parent_lock, (OUT/'runner.lock').open('a') as own_lock:
            fcntl.flock(parent_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(own_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            main()
    except Exception as exc:
        status(state='failed',error=repr(exc));raise
