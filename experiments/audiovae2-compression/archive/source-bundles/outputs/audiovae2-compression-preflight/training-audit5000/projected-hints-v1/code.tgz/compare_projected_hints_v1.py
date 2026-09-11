"""Matched recovery comparison with two external, trainable teacher readouts.

Starts from the retained accumulation12 candidate. Diagnostic source reuse is
explicit; this is not a continuation of the main training ledger.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import torch
import compare_accumulation_v1 as previous
import projected_hints_v1 as hints
from unified_monitor import UnifiedMonitor

replay,base,screen,resume=previous.replay,previous.base,previous.screen,previous.resume
VERSION='audiovae2_projected_hints_comparison_v1'
ARMS=('baseline','two_hints')
START=4625;SOURCES=1500;FRESH_START=10500;ACCUMULATION=12;EVERY=60


def restart_candidate(model,candidate):
    if candidate.get('format')!=previous.VERSION or candidate.get('optimizer_step')!=START or candidate.get('accumulation')!=12:
        raise ValueError('Expected the retained accumulation12 candidate')
    payload={'group':candidate['group'],'optimizer':candidate['optimizer'],'rng':candidate['rng'],
             'step':START,'identity':candidate['original_training_identity']}
    optimizer=resume.continuation.restore_training_state(model,payload)
    checks={k:replay.compare_tree(a,b) for k,a,b in (
        ('group',dict(model.group_state_dict()),dict(candidate['group'])),
        ('optimizer',optimizer.state_dict(),candidate['optimizer']),('rng',screen.rng_state(),candidate['rng']))}
    if not all(v['equal'] for v in checks.values()):raise RuntimeError('Common starting state did not restore exactly')
    return optimizer,checks


def copy_projectors(projectors,*,trainable):
    return {k:hints.LinearHint(p.weight,p.bias,trainable=trainable) for k,p in projectors.items()}


def projector_optimizer(projectors):
    return torch.optim.AdamW([p for q in projectors.values() for p in q.parameters()],lr=3e-5,betas=(.9,.99),eps=1e-8,weight_decay=0)


@torch.no_grad()
def evaluate_hints(model,teacher,crops,projectors,initial_projectors):
    total=sum(c['valid_scored_samples'] for c in crops)
    aggregate={label:{k:0. for k in hints.HINT_PATHS} for label in ('current','initial_frozen')};rows=[]
    for crop in crops:
        z,_,valid,_=base.batch([crop])
        with hints.capture_hints(teacher.model.decoder,detach=True) as tf:trace=base.teacher_forward(teacher,z)
        hints.contribution.warm_student(model,trace['group_input'])
        with hints.capture_hints(model.decoder,detach=True) as sf:model.group_from_input(trace['group_input'])
        row={'source_id':crop['source_id'],'valid_samples':int(valid.sum())}
        for label,maps in (('current',projectors),('initial_frozen',initial_projectors)):
            values=hints.hint_losses(maps,sf,tf,valid,total)
            for key,value in values.items():aggregate[label][key]+=float(value)
            row[label]={key:float(value)*total/max(int(valid.sum()),1) for key,value in values.items()}
        rows.append(row)
    return {'aggregate':aggregate,'rows':rows,'valid_samples':total,'source_ids':[c['source_id'] for c in crops]}


def authenticate_candidate(path,original_payload,chosen_ids):
    receipt=json.loads((path.parent/'completed.json').read_text());top=json.loads((path.parent.parent/'completed.json').read_text())
    checksum=base.sha(path)
    if (receipt.get('checkpoint_sha256')!=checksum or receipt.get('arm')!='accumulation12'
            or top.get('comparison_valid') is not True or top.get('arms',{}).get('accumulation12')!=receipt):
        raise ValueError('Retained candidate receipt changed')
    c=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (c.get('format')!=previous.VERSION or c.get('optimizer_step')!=START or c.get('accumulation')!=12
            or c.get('updates')!=125 or c.get('sources_consumed')!=SOURCES
            or c.get('additional_sources_seen')!=chosen_ids or c.get('historical_sources_seen')!=original_payload['sources_seen']
            or c.get('original_training_identity')!=original_payload['identity']):raise ValueError('Candidate recipe or source history changed')
    reference=json.loads((path.parent/'development-source1500.json').read_text())
    if reference['aggregate']!=receipt['final']:raise ValueError('Saved candidate quality receipt changed')
    return c,reference,checksum


def main():
    p=argparse.ArgumentParser()
    for n in ('checkpoint','candidate','anchor-checkpoint','screen-out','base-out','manifest','fresh-manifest','shards','assets','out'):
        p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Use a fresh isolated comparison directory')
    if shutil.disk_usage(args.out.parent).free<400<<20:raise OSError('Insufficient scratch space')
    base.policy()
    _,selection,_,manifest,pools,original,_,original_sha,_=resume.authenticate(args)
    data=replay.FreshTrainingData(args.fresh_manifest,args.manifest,pools,args.shards)
    ids=list(data.source_ids[FRESH_START:FRESH_START+SOURCES])
    if len(ids)!=SOURCES or len(set(ids))!=SOURCES:raise ValueError('Expected1500 unique diagnostic sources')
    if set(ids)&{c['source_id'] for c in pools['calibration']+pools['development']}:raise ValueError('Calibration/development leakage')
    for n in range(FRESH_START,FRESH_START+SOURCES,300):data.take(n,300)
    candidate,reference,candidate_sha=authenticate_candidate(args.candidate,original,ids)
    if (candidate['identity']['start_checkpoint_sha256']!=original_sha
            or candidate['identity']['runner_sha256']!=base.sha(previous.__file__)
            or candidate['identity']['resume_identity']!=original['resume_identity']
            or candidate['identity']['source_ids']!=ids):raise ValueError('Candidate original4500/code/source identity changed')
    calibration=pools['calibration'];development=pools['development']
    if len(calibration)!=72 or len(development)!=96:raise ValueError('Fixed calibration/development panels changed')
    hints.contribution.validate_fit_sources(calibration,development)
    by_id={c['source_id']:c for c in development};cases=[by_id[s] for s in replay.CASE_IDS]
    protected_paths=[args.checkpoint,args.candidate,args.anchor_checkpoint,args.manifest,args.fresh_manifest,
        args.candidate.parent/'completed.json',args.candidate.parent.parent/'completed.json',args.candidate.parent/'development-source1500.json',
        args.checkpoint.parent/'checkpoint-step5000.pt',args.checkpoint.parent/'checkpoint-step5000.json',
        Path(__file__),Path(hints.__file__),Path(previous.__file__),Path(base.__file__),Path(screen.__file__),Path(resume.__file__)]
    protected={str(path):base.sha(path) for path in protected_paths}
    teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    model=base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices'])
    common=base.objective();coefficients=candidate['original_training_identity']['coefficients']
    frozen=screen.frozen_versions(model);teacher_frozen=resume.continuation.teacher_versions(teacher)
    optimizer,restoration=restart_candidate(model,candidate)
    args.out.mkdir();started=time.monotonic();failure=None;results={};initial_first=None
    identity={'version':VERSION,'runner_sha256':base.sha(__file__),'helper_sha256':base.sha(hints.__file__),
        'candidate_sha256':candidate_sha,'original4500_sha256':original_sha,'starting_optimizer_step':START,
        'final_optimizer_step':START+SOURCES//ACCUMULATION,'sources_each':SOURCES,'source_interval':[FRESH_START,FRESH_START+SOURCES],
        'source_ids':ids,'source_ids_sha256':screen.digest(ids),'calibration_ids':[c['source_id'] for c in calibration],
        'development_ids':[c['source_id'] for c in development],'arms':list(ARMS),'coefficients':coefficients,
        'learning_rate':3e-5,'execution_batch_size':1,'gradient_accumulation':ACCUMULATION,
        'diagnostic_data_reuse':True,'automatic_promotion':False,'projector_training':'joint, separate fresh AdamW; student moments preserved',
        'hint_paths':hints.HINT_PATHS,'samples_per_feature_cell':hints.SAMPLES_PER_CELL,'projection_gradients_excluded_from_balance':True,
        'protected':protected,'backend':replay.backend_state(),'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version()}
    base.write_json(args.out/'launch.json',identity)
    try:
        with replay.diagnostic_state_guard(model,teacher,optimizer):
            projectors,fit=hints.fit_projectors(model,teacher,calibration,trainable=True)
            hint_coefficients,balance=hints.calibrate_hint_coefficients(model,teacher,calibration,common,coefficients,projectors,fraction=.05)
        initial_projectors=copy_projectors(projectors,trainable=False)
        base.write_json(args.out/'projector-fit.json',fit);base.write_json(args.out/'hint-balance.json',balance)
        torch.save({k:q.state_dict() for k,q in initial_projectors.items()},args.out/'initial-projectors.pt')
        identity['hint_coefficients']=hint_coefficients;base.write_json(args.out/'launch.json',identity)
        monitor=UnifiedMonitor(None,{},{})
        for arm in ARMS:
            folder=args.out/arm;folder.mkdir();optimizer,restoration=restart_candidate(model,candidate)
            maps=copy_projectors(initial_projectors,trainable=True)
            popt=projector_optimizer(maps) if arm=='two_hints' else None
            weights=hint_coefficients if arm=='two_hints' else {k:0. for k in hints.HINT_PATHS}
            with replay.diagnostic_state_guard(model,teacher,optimizer):
                initial=monitor.evaluate(model,teacher,development,common)
                initial_hints=evaluate_hints(model,teacher,development,maps,initial_projectors)
                initial_cases=replay.evaluate_cases(model,teacher,cases)
            saved_check=previous.numeric_reference_check(initial,reference)
            paired=saved_check if initial_first is None else previous.numeric_reference_check(initial,initial_first)
            base.write_json(folder/'restoration.json',{'state':restoration,'saved_quality':saved_check,'paired_quality':paired})
            if not saved_check['passed'] or not paired['passed']:raise RuntimeError('Common saved96-source starting quality changed')
            initial_first=initial if initial_first is None else initial_first
            base.write_json(folder/'development-source0.json',initial);base.write_json(folder/'hints-source0.json',initial_hints)
            snapshots=[{'sources_consumed':0,'optimizer_step':START,'cases':initial_cases}];seen=[];samples=0;cache_checks=[];warmed=set()
            for consumed in range(0,SOURCES,ACCUMULATION):
                crops=data.take(FRESH_START+consumed,ACCUMULATION);chosen=[c['source_id'] for c in crops]
                if chosen!=ids[consumed:consumed+ACCUMULATION]:raise ValueError('Source order changed')
                for crop in crops:
                    shape=tuple(crop['latents'].shape)
                    if shape not in warmed:
                        with replay.diagnostic_state_guard(model,teacher,optimizer),torch.no_grad():
                            z,_,_,_=base.batch([crop]);tr=base.teacher_forward(teacher,z)
                            hints.contribution.warm_student(model,tr['group_input'])
                        warmed.add(shape)
                with replay.observe_teacher_cache(crops) as checks:
                    values=hints.training_update(model,teacher,crops,common,coefficients,optimizer,maps,weights,
                        projector_optimizer=popt,record_diagnostics=True,record_components=consumed in (0,SOURCES-ACCUMULATION))
                cache_checks.extend(checks);seen.extend(chosen);samples+=sum(c['valid_scored_samples'] for c in crops)
                record={'sources_consumed':len(seen),'optimizer_step':START+len(seen)//ACCUMULATION,
                    'source_ids':chosen,'scored_samples':samples,'values':values,'teacher_cache_checks':checks}
                with (folder/'train.jsonl').open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')
                if len(seen)%EVERY==0:
                    with replay.diagnostic_state_guard(model,teacher,optimizer):measured=replay.evaluate_cases(model,teacher,cases)
                    snapshots.append({'sources_consumed':len(seen),'optimizer_step':record['optimizer_step'],'cases':measured})
                    base.write_json(folder/'case-trajectories.json',snapshots)
                    base.event('projected_hint_progress',arm=arm,sources=len(seen),optimizer_step=record['optimizer_step'])
            with replay.diagnostic_state_guard(model,teacher,optimizer):
                final=monitor.evaluate(model,teacher,development,common)
                final_hints=evaluate_hints(model,teacher,development,maps,initial_projectors)
            base.write_json(folder/'development-source1500.json',final);base.write_json(folder/'hints-source1500.json',final_hints)
            if seen!=ids or len(snapshots)!=26:raise RuntimeError('Incomplete matched exposure')
            if any(float(v['step'])!=START+125 for v in optimizer.state.values()):raise RuntimeError('Student AdamW counter differs')
            if screen.frozen_versions(model)!=frozen or resume.continuation.teacher_versions(teacher)!=teacher_frozen:raise RuntimeError('Frozen state changed')
            if set(model.group_state_dict())!=set(candidate['group']):raise RuntimeError('Projector entered decoder state')
            artifact={'format':VERSION,'arm':arm,'group':{k:v.detach().cpu() for k,v in model.group_state_dict().items()},
                'optimizer':optimizer.state_dict(),'rng':screen.rng_state(),'optimizer_step':START+125,
                'projectors':{k:q.state_dict() for k,q in maps.items()},'projector_optimizer':None if popt is None else popt.state_dict(),
                'diagnostic_sources_seen':seen,'identity':identity,'automatic_promotion':False}
            torch.save(artifact,folder/'final.pt')
            movement={k:float((maps[k].weight-initial_projectors[k].weight).double().norm()) for k in hints.HINT_PATHS}
            result={'arm':arm,'initial':initial['aggregate'],'final':final['aggregate'],
                'initial_near':{k:v for k,v in initial['overview_window_metrics'].items() if k!='by_source'},
                'final_near':{k:v for k,v in final['overview_window_metrics'].items() if k!='by_source'},
                'trajectory':previous.trajectory_summary(snapshots),'hint_initial':initial_hints['aggregate'],'hint_final':final_hints['aggregate'],
                'projector_weight_movement_norm':movement,'sources':len(seen),'scored_samples':samples,'updates':125,'optimizer_step':START+125,
                'checkpoint_sha256':base.sha(folder/'final.pt'),'teacher_cache_all_bitwise':all(c['bitwise_equal'] for c in cache_checks),
                'teacher_cache_all_original_tolerance':all(c['allclose_original_tolerance'] for c in cache_checks),'frozen_state_preserved':True,
                'saved_initial_quality_passed':saved_check['passed'],'paired_initial_quality_passed':paired['passed'],'automatic_promotion':False}
            base.write_json(folder/'completed.json',result);results[arm]=result
            data.assert_unchanged()
            if any(base.sha(path)!=value for path,value in protected.items()):raise RuntimeError('Original protected files changed')
        if results['baseline']['scored_samples']!=results['two_hints']['scored_samples']:raise RuntimeError('Exposure differs')
        base.write_json(args.out/'completed.json',{'version':VERSION,'arms':results,'matched_exposure':True,'elapsed_seconds':time.monotonic()-started,
            'comparison_valid':all(r['teacher_cache_all_original_tolerance'] and r['saved_initial_quality_passed'] and r['paired_initial_quality_passed'] for r in results.values()),
            'original_files_preserved':True,'automatic_promotion':False,'main_training_unchanged':True})
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'completed_arms':list(results)};base.write_json(args.out/'failure.json',failure);raise
    finally:
        base.write_json(args.out/'preservation.json',{'failure':failure,'frozen_student_preserved':screen.frozen_versions(model)==frozen,
            'teacher_preserved':resume.continuation.teacher_versions(teacher)==teacher_frozen,
            'original_files_preserved':all(base.sha(path)==value for path,value in protected.items())})

if __name__=='__main__':main()
