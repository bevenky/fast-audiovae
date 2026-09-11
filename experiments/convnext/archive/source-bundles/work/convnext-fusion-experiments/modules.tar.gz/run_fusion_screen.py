"""Bounded, paired decoder refinement screen on the preserved step-8090 parent."""
from __future__ import annotations
import argparse, copy, gc, hashlib, json, os, shutil, time
from dataclasses import asdict, replace
from pathlib import Path
import torch
from audiovae_student.cache import TrainingCrop, sample_training_crop
from audiovae_student.continuation_data import load_continuation_plan
from audiovae_student.data import load_manifest
from audiovae_student.prepare_targets import _target_teacher
from audiovae_student.teacher import FrozenAudioVAE2
from audiovae_student.source_corpus import SourceCorpus
from audiovae_student.representative_pilot import _record
from audiovae_student.preflight_distillation import _crop_identity
from audiovae_student.objective_comparison import state_fingerprint
from audiovae_student.restart_data import file_sha, digest
from audiovae_student.training import _restore_rng, _rng_state
from audiovae_student.fusion_migration import build_fusion_engine
from audiovae_student.fusion_evaluation import evaluate_fusion, diagnose_streaming
from audiovae_student.recipe_v2 import scored_batch_v2
from audiovae_student.discriminators import discriminator_loss
BASE=Path('/workspace/fast-audiovae-convnext-20260909-r9')
OLD=Path('/workspace/fast-audiovae-convnext-20260908-r1')
OUT=BASE/'remediation/fusion-screen'
ARMS=('control','tanh','short_mel','filter','fresh_magnitude','complex')
STEPS=200
WARMUP=20
BATCH=32


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temp,path)


def status(**value):
    value['time_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    atomic_json(OUT/'experiment-status.json',value)
    print(json.dumps(value),flush=True)


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def save(path,value):
    if shutil.disk_usage(OUT).free < 1500*1024**2:
        raise RuntimeError('Insufficient checkpoint reserve; original parent retained')
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(cpu_tree(value),f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def panels(parent,teacher):
    all_crops=[]; metadata={}; hashes={}
    specs=[(BASE/'heldout-targets.pt',parent['identity']['heldout_metadata'])]
    for name in ['validation-appendix','language-validation']:
        root=BASE/'remediation'/name
        specs.append((root/'heldout-targets.pt',json.loads((root/'groups.json').read_text())))
    for path,groups in specs:
        hashes[str(path)]=file_sha(path)
        if path==BASE/'heldout-targets.pt':
            assert hashes[str(path)]==parent['identity']['data']['heldout_targets']['sha256']
        else:
            descriptor=json.loads(path.with_suffix('.json').read_text())
            # Descriptor schemas are pinned below by their own hashes too.
            hashes[str(path.with_suffix('.json'))]=file_sha(path.with_suffix('.json'))
            expected=descriptor.get('sha256') or descriptor.get('file_sha256')
            if expected is not None:assert hashes[str(path)]==expected
        saved=torch.load(path,map_location='cpu',weights_only=True)
        crops=[TrainingCrop(**x) for x in saved['crops']]
        all_crops.extend(crops);metadata.update(groups)
    # Genuine silence and low-level input are encoded by the same frozen encoder.
    generator=torch.Generator().manual_seed(98213)
    zero=torch.zeros(1,1,96000)
    fixtures={'encoded_zero':zero,'encoded_quiet_noise':torch.randn(1,1,96000,generator=generator)*1e-5,
              'encoded_fade':torch.sin(torch.arange(96000).float()[None,None,:]*(2*torch.pi*220/16000))*.01*torch.linspace(1,0,96000)[None,None,:]}
    frozen=[]
    with torch.no_grad():
        for name,audio in fixtures.items():
            z=teacher.encode(audio.cuda());target=teacher.decode(z)
            z=z.detach().cpu();target=target.detach().cpu()
            crop=TrainingCrop(z,target,audio,name,name,0,0,0,z.shape[-1],target.shape[-1])
            all_crops.append(crop);frozen.append(asdict(crop));metadata[name]={'dataset':'synthetic_fixture','event':name,'language':'none','speech':False,'condition':name}
    save(OUT/'synthetic-targets.pt',{'crops':frozen})
    keys=[(c.source_id,c.start_frame) for c in all_crops]
    assert len(keys)==len(set(keys)), 'Duplicate heldout crops'
    return all_crops,metadata,{'source_hashes':hashes,'crops':_crop_identity(all_crops),'count':len(all_crops)}


def prepare(parent,receipt):
    plan=load_continuation_plan(Path(receipt['plan_path']))
    cursor=parent['sampler']['cursor']
    windows=plan['windows'][cursor:cursor+(STEPS+WARMUP)*BATCH]
    assert len(windows)==(STEPS+WARMUP)*BATCH
    rows={r.source_id:r for r in plan['rows']}
    # Overlap was excluded in the sealed plan; audit this exact subset again.
    ranges={}
    for w in windows:
        interval=(w.start_frame*1920,w.start_frame*1920+w.valid_output_samples48k)
        ranges.setdefault(w.source_id,[]).append(interval)
    for intervals in ranges.values():
        intervals.sort()
        assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])), 'Repeated scored audio inside an arm'
    teacher=FrozenAudioVAE2.from_files(OLD/'assets/audio_vae_v2.py',OLD/'assets/audiovae.pth',device='cuda')
    first=load_manifest('/workspace/fast-audiovae-convnext-20260909-r2/data/selection-v1/train-manifest.jsonl')[0]
    teacher=_target_teacher(teacher,first)
    before=state_fingerprint(teacher.model.state_dict())
    assert before==parent['identity']['data']['teacher_state_sha256']
    heldout,metadata,panel_identity=panels(parent,teacher)
    assert not (set(ranges)&{c.source_id for c in heldout}), 'Heldout source selected for training'
    wanted={}
    for i,w in enumerate(windows):wanted.setdefault(w.source_id,[]).append((i,w))
    crops=[None]*len(windows)
    # All selected crop tensors remain in RAM and are shared across arms; no data copies per experiment.
    corpus=SourceCorpus(plan['rows'],teacher,cache_dir=OUT/'teacher-cache',
        input_sample_counts=plan['counts'],max_disk_bytes=256*1024**2,
        min_free_bytes=2*1024**3,max_memory_utterances=16,allow_prepared_source=True)
    sources=list(wanted)
    try:
        for offset in range(0,len(sources),8):
            ids=sources[offset:offset+8]
            corpus.prefetch(ids,max_batch_size=8,max_total_input_samples=1_920_000)
            for sid in ids:
                record=_record(corpus,rows[sid],plan['counts'][sid])
                for index,w in wanted[sid]:
                    crop=sample_training_crop(record,w.start_frame,w.scored_frames,context_frames=30)
                    crops[index]=replace(crop,valid_scored_samples=w.valid_output_samples48k)
            if offset%160==0:status(state='preparing_frozen_targets',prepared_sources=min(offset+8,len(sources)),total_sources=len(sources))
    finally:corpus.close()
    assert state_fingerprint(teacher.model.state_dict())==before
    assert all(c is not None for c in crops)
    teacher_identity=teacher.provenance
    del teacher;gc.collect();torch.cuda.empty_cache()
    identity={'parent_sha256':receipt['checkpoint_sha256'],'parent_step':parent['engine']['step'],
        'parent_cursor':cursor,'generator_steps':STEPS,'batch_size':BATCH,'discriminator_warmup_steps':WARMUP,
        'arms':list(ARMS),'training_windows':[w.to_dict() for w in windows],
        'teacher':teacher_identity,'teacher_state_sha256':before,'target_crops':_crop_identity(crops),
        'heldout':panel_identity,'context_frames':30,'rng_policy':'parent per arm; independent MRD seed 93508 and warmup seed 31904',
        'script_sha256':file_sha(Path(__file__)),
        'source_sha256':{p.name:file_sha(p) for p in (Path(__file__).parent/'audiovae_student').glob('*.py')},
        'data_repetition':'none within each arm; deliberately matched across independent debugging arms'}
    atomic_json(OUT/'experiment-identity.json',identity)
    return crops[:STEPS*BATCH],crops[STEPS*BATCH:],heldout,metadata,identity


def warmup(engine,crops,arm):
    original_rng=engine.crop_generator.get_state()
    engine.crop_generator.manual_seed(31904)
    baseline=state_fingerprint(engine.model.state_dict())
    for i in range(WARMUP):
        batch_crops=crops[i*BATCH:(i+1)*BATCH]
        engine.model.eval()
        with torch.no_grad():
            batch,_=scored_batch_v2(engine.model,batch_crops,engine.reconstruction,engine.device)
            predicted,target,count=engine._perceptual_audio(batch)
        assert count==BATCH
        engine.discriminators.train()
        engine.discriminator_optimizer.zero_grad(set_to_none=True)
        loss=discriminator_loss(engine.discriminators,predicted,target,
            example_weights=predicted.new_tensor([c.valid_scored_samples for c in batch_crops]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.discriminators.parameters(),engine.config.gradient_clip,error_if_nonfinite=True)
        engine.discriminator_optimizer.step()
        engine.discriminator_optimizer.zero_grad(set_to_none=True)
    engine.crop_generator.set_state(original_rng)
    assert state_fingerprint(engine.model.state_dict())==baseline
    status(state='discriminator_warmup_complete',arm=arm,updates=WARMUP,last_loss=float(loss.detach()))


def cpu_diagnostics(parent,heldout):
    # Bounded engineering screen of unfused candidate heads, not a deployed-kernel RTF claim.
    z=next(c.latents[...,:50].clone() for c in heldout if c.latents.shape[-1]>=50)
    engines={name:build_fusion_engine(parent['engine'],name,device='cpu')[0]
             for name in ['control','tanh','filter','zero_padding']}
    out={}
    for name,engine in engines.items():
        out[name]=diagnose_streaming(engine.model,z,frames_per_chunk=2)
    timings={name:[] for name in engines}
    generator=torch.Generator().manual_seed(603)
    with torch.inference_mode():
        for engine in engines.values():
            engine.model.eval();engine.model(z)
        for _ in range(7):
            order=torch.randperm(len(engines),generator=generator).tolist()
            for index in order:
                name=list(engines)[index];model=engines[name].model
                start=time.perf_counter();state=model.initial_state();count=0
                for chunk in z.split(2,dim=-1):
                    y,state=model.forward_stream(chunk,state);count+=y.shape[-1]
                elapsed=time.perf_counter()-start
                assert count==z.shape[-1]*1920
                timings[name].append(elapsed/(count/48000))
    import statistics
    for name,values in timings.items():out[name]['one_thread_unfused_stream_rtf_median']=statistics.median(values);out[name]['rtf_samples']=values
    base=engines['control'].model.eval();zero=engines['zero_padding'].model.eval()
    startup=[]
    with torch.inference_mode():
        for crop in heldout:
            if crop.context_start_frame!=0:continue
            x=crop.latents
            a=base(x);b=zero(x)
            total=min(a.shape[-1],crop.teacher_audio.shape[-1]);boundary=min(total,29*1920)
            target=crop.teacher_audio[...,:total]
            row={'source_id':crop.source_id,'start_frame':crop.start_frame,
                'startup_samples':boundary,'base_startup_mae':float((a[...,:boundary]-target[...,:boundary]).abs().mean()),
                'zero_startup_mae':float((b[...,:boundary]-target[...,:boundary]).abs().mean())}
            if total>boundary:row['mature_max_difference']=float((a[...,boundary:total]-b[...,boundary:total]).abs().max())
            startup.append(row)
    out['zero_padding_startup']=startup
    atomic_json(OUT/'cpu-diagnostics.json',out)
    del engines;gc.collect()
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',choices=ARMS);parser.add_argument('--skip-completed',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    receipt=json.loads((OUT/'parent-receipt.json').read_text())
    assert file_sha(OUT/'parent.pt')==receipt['checkpoint_sha256']
    parent=torch.load(OUT/'parent.pt',map_location='cpu',weights_only=True)
    status(state='preparing',parent_step=parent['engine']['step'])
    training,d_warmup,heldout,metadata,identity=prepare(parent,receipt)
    # CPU diagnostics after teacher preparation, with no concurrent GPU trainer.
    if not (OUT/'cpu-diagnostics.json').exists():cpu_diagnostics(parent,heldout)
    for arm in ([args.only] if args.only else ARMS):
        directory=OUT/arm
        if (directory/'complete.json').exists() and args.skip_completed:continue
        if directory.exists():raise RuntimeError('Refusing to replay an incomplete arm: '+arm)
        directory.mkdir()
        _restore_rng(parent['rng']);torch.manual_seed(93508)
        engine,migration=build_fusion_engine(parent['engine'],arm,device='cuda')
        _restore_rng(parent['rng'])
        atomic_json(directory/'migration.json',migration)
        status(state='evaluating_before',arm=arm)
        before=evaluate_fusion(engine,heldout,metadata)
        atomic_json(directory/'before.json',before)
        _restore_rng(parent['rng'])
        if arm in ['fresh_magnitude','complex']:warmup(engine,d_warmup,arm)
        started=time.perf_counter()
        for i in range(STEPS):
            batch=training[i*BATCH:(i+1)*BATCH]
            atomic_json(directory/'inflight.json',{'update':i+1,'crop_ids':[(c.source_id,c.start_frame) for c in batch]})
            metrics=engine.train_step(batch)
            with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
            (directory/'inflight.json').unlink()
            if (i+1)%25==0:status(state='training',arm=arm,completed_updates=i+1,total_updates=STEPS,seconds=time.perf_counter()-started,
                waveform=metrics['teacher_waveform'],mel=metrics['teacher_mel'])
        seconds=time.perf_counter()-started
        status(state='evaluating_after',arm=arm,training_seconds=seconds)
        after=evaluate_fusion(engine,heldout,metadata)
        atomic_json(directory/'after.json',after)
        checkpoint={'format_version':'fusion_screen_v1','engine':engine.state_dict(),'rng':_rng_state(),
            'experiment_identity_sha256':digest(identity),'parent_sha256':receipt['checkpoint_sha256'],
            'variant':arm,'migration':migration,'generator_updates':STEPS,'discriminator_only_updates':WARMUP if arm in ['fresh_magnitude','complex'] else 0,
            'training_target_identity_sha256':digest(identity['target_crops']),'seconds':seconds}
        save(directory/'final.pt',checkpoint)
        atomic_json(directory/'complete.json',{'arm':arm,'updates':STEPS,'step':engine.step,'seconds':seconds,
            'checkpoint_sha256':file_sha(directory/'final.pt'),'before_summary':before['summary'],'after_summary':after['summary']})
        del engine,checkpoint;gc.collect();torch.cuda.empty_cache()
    assert file_sha(OUT/'parent.pt')==receipt['checkpoint_sha256']
    assert file_sha(Path(receipt['source_checkpoint']))==receipt['checkpoint_sha256']
    status(state='complete',arms=[x for x in ARMS if (OUT/x/'complete.json').exists()],original_run_still_paused=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        status(state='failed',error=repr(error));raise
