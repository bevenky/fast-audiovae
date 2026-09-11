"""Read-only paired quiet-window diagnosis on the fixed96-source panel.

No fitting, optimizer, intervention, resampling, or modified acceptance limit.
The original model/metric passes are observed; source16k zeros are compared with
teacher48k quiet windows without pretending they share a waveform sampling grid.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
import torch
import joint_recovery_v1 as joint
import diagnose_channel_contributions_v2 as common
from unified_monitor import UnifiedMonitor

base,screen,replay,resume=joint.base,joint.screen,joint.replay,joint.resume
VERSION='audiovae2_quiet_window_audit_v1'
PERIODS=(2,4,8,40,240,1920)


def source_reference_window(crop,a,b):
    """Map a scored48k span to overlapping original16k cells, without resampling."""
    if type(a)is not int or type(b)is not int or a<0 or b<=a:raise ValueError('Invalid scored window extent')
    reference=crop['reference16k'].detach().cpu().double().reshape(-1)
    start=crop['context_frames']*1920+a;stop=crop['context_frames']*1920+b
    first=start//3;last=(stop+2)//3
    if last>reference.numel():raise ValueError('Reference window exceeds authenticated source crop')
    values=reference[first:last];indices=torch.arange(first,last,dtype=torch.int64)
    weights=(torch.minimum(3*indices+3,torch.tensor(stop))-torch.maximum(3*indices,torch.tensor(start))).clamp_min(0).double()
    if float(weights.sum())!=b-a or not torch.isfinite(values).all():raise ValueError('Invalid reference alignment')
    return {'source_reference_rms':float(((values.square()*weights).sum()/weights.sum()).sqrt()),
        'source_reference_exact_zero':bool((values==0).all()),'source_reference_samples':values.numel(),
        'source_reference_weighted_48k_samples':int(weights.sum()),'source_reference_nonzero_samples':int((values!=0).sum()),
        'source_reference_window_16k':[first,last],
        'source_reference_alignment':'Original16k cells overlap3 corresponding48k sample slots; weighted partial endpoints, no resampling'}


def teacher_bin(rms):
    if rms==0:return 'exact_zero'
    if rms<=1e-5:return 'nonzero_to_1e-5'
    if rms<=1e-4:return '1e-5_to_1e-4'
    return '1e-4_to_1e-3'


def temporal_bin(start_sample):
    if start_sample<960:return 'source_first20ms'
    if start_sample<1920:return 'source20_to40ms'
    if start_sample<38400:return 'source40_to800ms'
    return 'source_after800ms'


def describe_quiet_windows(prediction,target,crop,raw):
    if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[:2]!=(1,1):raise ValueError('Expected singleton scored waveform')
    p,t=prediction.detach().cpu().double().reshape(-1),target.detach().cpu().double().reshape(-1)
    rows=[]
    for i,original in enumerate(raw['windows']):
        if not original['is_quiet']:continue
        row=dict(original);a,b=row['start_sample'],row['stop_sample'];pp,tt=p[a:b],t[a:b]
        if len(pp)!=row['valid_samples']:raise ValueError('Expected the unchanged contiguous scored crop grid')
        e=pp-tt;dc=float(e.mean());energy=float(e.square().sum());centered=e-dc
        penergy=float(pp.square().sum());tenergy=float(tt.square().sum());dot=float((pp*tt).sum())
        ep=pp-pp.mean();et=tt-tt.mean();cden=float(ep.norm()*et.norm());den=math.sqrt(penergy*tenergy)
        rf=row['residual_rms']>row['residual_limit'];af=row['student_rms']>row['output_rms_limit']
        absolute=crop['start_frame']*1920+a;maximum=int(e.abs().argmax())
        row.update(source_id=crop['source_id'],window_id=crop['source_id']+':'+str(absolute),
            source_start_sample=absolute,source_stop_sample=crop['start_frame']*1920+b,
            teacher_level_bin=teacher_bin(row['teacher_rms']),temporal_bin=temporal_bin(absolute),
            source_starting_crop=crop['context_frames']==0 and crop['start_frame']==0,
            actual_startup=crop['context_frames']==0 and crop['start_frame']==0 and absolute<1920,
            context_frames=crop['context_frames'],partial_window=row['valid_samples']!=960,
            adjacent_active_window=any(not raw['windows'][j]['is_quiet'] for j in (i-1,i+1) if 0<=j<len(raw['windows'])),
            failure_category='both' if rf and af else 'residual_only' if rf else 'amplitude_only' if af else 'passed',
            residual_over_limit=row['residual_rms']/row['residual_limit'],output_rms_over_limit=row['student_rms']/row['output_rms_limit'],
            residual_limit_margin=row['residual_limit']-row['residual_rms'],output_limit_margin=row['output_rms_limit']-row['student_rms'],
            prediction_mean=float(pp.mean()),teacher_mean=float(tt.mean()),residual_mean=dc,
            residual_square_sum=energy,residual_rms_double=math.sqrt(energy/len(e)),
            centered_residual_square_sum=float(centered.square().sum()),centered_residual_rms=float(centered.square().mean().sqrt()),
            dc_energy_fraction=dc*dc*len(e)/energy if energy else 0.,
            least_squares_gain=dot/tenergy if tenergy else None,cosine=dot/den if den else None,
            centered_cosine=float((ep*et).sum())/cden if cden else None,
            teacher_square_sum=tenergy,prediction_square_sum=penergy,dot_sum=dot,
            residual_abs_max=float(e.abs().max()),maximum_error_source_sample=absolute+maximum,
            maximum_error_signed_residual=float(e[maximum]),maximum_error_prediction=float(pp[maximum]),maximum_error_teacher=float(tt[maximum]))
        row.update(source_reference_window(crop,a,b));rows.append(row)
    return rows


def phase_templates(prediction,target,crop,raw,periods=PERIODS,min_cycles=8):
    """Contiguous quiet runs only; absolute source-grid phase, no sample joining."""
    if type(min_cycles)is not int or min_cycles<8:raise ValueError('At least8 complete cycles required')
    if any(type(p)is not int or p<=0 for p in periods):raise ValueError('Invalid phase period')
    residual=(prediction.detach().cpu().double()-target.detach().cpu().double()).reshape(-1)
    runs=[]
    for w in raw['windows']:
        if not w['is_quiet']:continue
        if runs and runs[-1][1]==w['start_sample']:runs[-1][1]=w['stop_sample']
        else:runs.append([w['start_sample'],w['stop_sample']])
    result=[];origin=crop['start_frame']*1920
    for start,stop in runs:
        for period in periods:
            a=((origin+start+period-1)//period)*period-origin;b=((origin+stop)//period)*period-origin
            cycles=max(0,(b-a)//period)
            row={'source_id':crop['source_id'],'quiet_run_scored_samples':[start,stop],'period_samples':period,
                 'aligned_source_samples':[origin+a,origin+b],'complete_cycles':cycles,'eligible':cycles>=min_cycles,
                 'excluded_run_samples':stop-start-max(0,b-a)}
            if row['eligible']:
                e=residual[a:b].reshape(cycles,period);template=e.mean(0);dc=float(e.mean());power=float(e.square().mean())
                ac=float((template-dc).square().mean());total_centered=float((e-dc).square().mean())
                row.update(residual_rms=math.sqrt(power),residual_dc=dc,phase_template_ac_rms=math.sqrt(ac),
                    phase_template_ac_fraction_of_residual_power=ac/power if power else 0.,
                    phase_template_ac_fraction_of_centered_power=ac/total_centered if total_centered else 0.,
                    phase_template_means=template.tolist() if period<=8 else None)
            result.append(row)
    return result


def summarize_windows(rows):
    counts={k:sum(r['failure_category']==k for r in rows) for k in ('passed','residual_only','amplitude_only','both')}
    n=sum(r['valid_samples'] for r in rows);e=sum(r['residual_square_sum'] for r in rows);ac=sum(r['centered_residual_square_sum'] for r in rows)
    return {'windows':len(rows),'samples':n,'failure_categories':counts,'failed':len(rows)-counts['passed'],
        'residual_rms':math.sqrt(e/n) if n else None,'centered_residual_rms':math.sqrt(ac/n) if n else None,
        'window_dc_energy_fraction':(e-ac)/e if e else 0.,'reference_exact_zero_windows':sum(r['source_reference_exact_zero'] for r in rows),
        'source_ids':sorted({r['source_id'] for r in rows})}


def grouped_summary(rows):
    result={'all':summarize_windows(rows)}
    for key in ('teacher_level_bin','temporal_bin','source_reference_exact_zero','adjacent_active_window','partial_window','actual_startup','source_starting_crop'):
        result[key]={str(value):summarize_windows([r for r in rows if r[key]==value]) for value in sorted({r[key] for r in rows})}
    result['by_source']={s:summarize_windows([r for r in rows if r['source_id']==s]) for s in sorted({r['source_id'] for r in rows})}
    return result


@torch.no_grad()
def evaluate_detailed(model,teacher,crops,spectral):
    original_quiet=base.quiet_window_metrics;original_teacher=base.teacher_forward
    rows=[];phases=[];cache=[];source_cursor=0;warmups={'teacher_extra_forwards':0,'student_extra_group_and_suffix_forwards':0}
    def observe(p,t,valid=None,**kwargs):
        nonlocal source_cursor
        if kwargs or (valid is not None and (valid.shape!=p.shape or valid.dtype!=torch.bool or not bool(valid.all()))):raise ValueError('Expected unchanged default quiet thresholds and full contiguous scored-window mask')
        raw=original_quiet(p,t,valid,**kwargs);crop=crops[source_cursor]
        if p.numel()!=crop['valid_scored_samples']:raise RuntimeError('Window observer source/extent differs')
        rows.extend(describe_quiet_windows(p,t,crop,raw));phases.extend(phase_templates(p,t,crop,raw));source_cursor+=1
        return raw
    def teacher_observer(t,z):
        warmed_before=len(getattr(t,'_compression_warmed_shapes',set()))
        trace=original_teacher(t,z)
        warmups['teacher_extra_forwards']+=3*(len(getattr(t,'_compression_warmed_shapes',set()))-warmed_before)
        crop=crops[len(cache)];a=crop['context_frames']*1920;b=a+crop['valid_scored_samples']
        target=crop['teacher_audio'][...,a:b].to(z.device);live=trace['waveform'][...,a:b]
        check={'source_id':crop['source_id'],'bitwise_equal':torch.equal(live,target),'max_abs':float((live-target).abs().max()),
               'passed':torch.allclose(live,target,atol=1e-5,rtol=1e-4)}
        cache.append(check)
        if not check['passed']:raise RuntimeError('Pristine teacher/cache mismatch')
        warmed_before=len(getattr(model,'_contribution_warmed_shapes',set()))
        common.warm_student(model,trace['group_input'])
        warmups['student_extra_group_and_suffix_forwards']+=3*(len(getattr(model,'_contribution_warmed_shapes',set()))-warmed_before)
        return trace
    base.quiet_window_metrics=observe;base.teacher_forward=teacher_observer
    try:report=joint.evaluate_review(UnifiedMonitor(None,{},{}),model,teacher,crops,spectral)
    finally:base.quiet_window_metrics=original_quiet;base.teacher_forward=original_teacher
    if source_cursor!=len(crops) or len(cache)!=len(crops):raise RuntimeError('Incomplete synchronous quiet capture')
    summary=grouped_summary(rows)
    if summary['all']['windows']!=report['aggregate']['quiet_windows'] or summary['all']['failed']!=report['aggregate']['quiet_failed_windows']:
        raise RuntimeError('Raw quiet rows differ from unchanged evaluation')
    return {'quality':report,'summary':summary,'windows':rows,'phase_templates':phases,'teacher_cache_checks':cache,'runtime_warmups':warmups,'scored_source_forwards':len(crops)}


def authenticate_final(args,candidate,data,pools):
    path=args.joint_run/'checkpoint-step5625.pt';checksum=base.sha(path)
    receipt=json.loads(path.with_suffix('.json').read_text());completion=json.loads((args.joint_run/'completed.json').read_text())
    launch=json.loads((args.joint_run/'launch.json').read_text());payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    expected_history=candidate['historical_sources_seen']+candidate['additional_sources_seen'];expected_added=list(data.source_ids[12000:24000])
    if (checksum!='0da2f6e98a29b025b41df84dbb674e1a35fbea2629f698ef33fe62f687c18b1b'
            or receipt.get('checkpoint_sha256')!=checksum or receipt.get('optimizer_step')!=5625
            or receipt.get('frozen_state_preserved') is not True or completion.get('status')!='completed_budget'
            or completion.get('step')!=5625 or completion.get('updates')!=1000 or completion.get('sources')!=12000
            or payload.get('format')!=joint.VERSION or payload.get('optimizer_step')!=5625
            or payload.get('updates')!=1000 or payload.get('accumulation')!=12
            or payload.get('historical_sources_seen')!=expected_history or payload.get('additional_sources_seen')!=expected_added
            or payload.get('identity')!=launch or payload.get('original_training_identity')!=candidate['original_training_identity']
            or launch.get('candidate_sha256')!=base.sha(args.candidate)
            or launch.get('source_interval')!=[12000,24000] or launch.get('source_ids_sha256')!=screen.digest(expected_added)):
        raise ValueError('Final joint checkpoint identity differs')
    if set(expected_added)&{c['source_id'] for c in pools['development']+pools['calibration']}:
        raise ValueError('Final checkpoint source lineage leaked held-out data')
    if any(base.sha(p)!=v for p,v in launch['protected'].items()):raise ValueError('Original joint protected source/input changed')
    reference=json.loads((args.joint_run/'development-step5625.json').read_text())
    if reference['aggregate']!=receipt['quality'] or reference['aggregate']!=completion['final']:raise ValueError('Final quality receipt differs')
    return payload,reference,checksum


def pair_windows(first,last):
    a={r['window_id']:r for r in first};b={r['window_id']:r for r in last}
    if len(a)!=len(first) or len(b)!=len(last) or set(a)!=set(b):raise RuntimeError('Paired quiet window identities differ')
    transition=Counter();rows=[]
    fixed=('source_id','valid_samples','source_start_sample','source_stop_sample','teacher_rms','teacher_level_bin',
           'residual_limit','output_rms_limit','source_reference_rms','source_reference_exact_zero')
    for key,x in a.items():
        y=b[key]
        if any(x[k]!=y[k] for k in fixed):raise RuntimeError('Teacher/source thresholds or quiet grid changed across checkpoints')
        transition[x['failure_category']+' -> '+y['failure_category']]+=1
        rows.append({'window_id':key,'source_id':x['source_id'],'initial_category':x['failure_category'],'final_category':y['failure_category'],
            'initial_residual_rms':x['residual_rms'],'final_residual_rms':y['residual_rms'],
            'residual_rms_change':y['residual_rms']-x['residual_rms'],'teacher_level_bin':x['teacher_level_bin'],
            'source_reference_exact_zero':x['source_reference_exact_zero'],'temporal_bin':x['temporal_bin']})
    return {'windows':len(rows),'category_transitions':dict(transition),'rows':rows}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','candidate','anchor-checkpoint','screen-out','base-out','manifest','fresh-manifest','shards','assets','joint-run','out'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Use a new isolated quiet audit directory')
    base.policy()
    _,selection,_,_,pools,original,_,original_sha,_=resume.authenticate(args)
    data=replay.FreshTrainingData(args.fresh_manifest,args.manifest,pools,args.shards)
    candidate,_,candidate_sha=joint.prior.authenticate_candidate(args.candidate,original,list(data.source_ids[10500:12000]))
    if candidate['identity']['start_checkpoint_sha256']!=original_sha:raise ValueError('Original candidate anchor differs')
    final,reference,final_sha=authenticate_final(args,candidate,data,pools)
    start_reference=json.loads((args.joint_run/'development-step4625.json').read_text());development=pools['development']
    if len(development)!=96 or [r['source_id'] for r in reference['rows']]!=[r['source_id'] for r in development]:raise ValueError('Expected original96-source held-out panel')
    protected_paths=[args.checkpoint,args.candidate,args.anchor_checkpoint,args.manifest,args.fresh_manifest,
        args.joint_run/'checkpoint-step5625.pt',args.joint_run/'checkpoint-step5625.json',args.joint_run/'completed.json',
        args.joint_run/'launch.json',args.joint_run/'development-step4625.json',args.joint_run/'development-step5625.json',
        args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',Path(__file__),Path(joint.__file__),Path(common.__file__),Path(base.__file__)]
    protected={str(path):base.sha(path) for path in protected_paths}
    teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    model=base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices']);spectral=base.objective()
    frozen=screen.frozen_versions(model);teacher_state=resume.continuation.teacher_versions(teacher)
    args.out.mkdir();started=time.monotonic();failure=None;results={}
    identity={'version':VERSION,'runner_sha256':base.sha(__file__),'initial_checkpoint_sha256':candidate_sha,'final_checkpoint_sha256':final_sha,
        'source_ids':[c['source_id'] for c in development],'source_ids_sha256':screen.digest([c['source_id'] for c in development]),
        'periods_samples':list(PERIODS),'minimum_contiguous_phase_cycles':8,'protected':protected,'backend':replay.backend_state(),
        'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'no_training':True,'no_model_intervention':True,
        'phase_limit':'Descriptive in-sample absolute-grid templates, not proof of a tone or a causal layer. Finite-cycle phase averaging captures some random variance (roughly1/cycles), even for unstructured noise.',
        'threshold_limit':'Unchanged provisional engineering checks, not calibrated audibility thresholds.',
        'source_reference_limit':'Reference16k RMS and exact zeros, no48k waveform resampling/comparison.',
        'score_scope':'One scored96-source pass per checkpoint plus separately counted original runtime warmups.'}
    base.write_json(args.out/'launch.json',identity)
    try:
        for step,payload,saved in ((4625,candidate,start_reference),(5625,final,reference)):
            model.load_group_state_dict(payload['group']);group_before={k:v.detach().cpu().clone() for k,v in model.group_state_dict().items()}
            with replay.diagnostic_state_guard(model,teacher):report=evaluate_detailed(model,teacher,development,spectral)
            check=joint.prior.previous.numeric_reference_check(report['quality'],saved)
            report['saved_quality_check']=check;base.write_json(args.out/f'quiet-step{step}.json',report)
            if not check['passed']:raise RuntimeError('Unchanged saved quality metrics did not reproduce')
            if not replay.compare_tree(dict(model.group_state_dict()),group_before)['equal']:raise RuntimeError('Read-only diagnostic altered model')
            results[step]=report;base.event('quiet_audit_progress',step=step,quiet=report['summary']['all'],seconds=time.monotonic()-started)
        paired=pair_windows(results[4625]['windows'],results[5625]['windows']);base.write_json(args.out/'paired-windows.json',paired)
        if screen.frozen_versions(model)!=frozen or resume.continuation.teacher_versions(teacher)!=teacher_state:raise RuntimeError('Frozen state changed')
        after_hashes={p:base.sha(p) for p in protected}
        if after_hashes!=protected:raise RuntimeError('Original protected file changed')
        base.write_json(args.out/'completed.json',{'version':VERSION,'complete':True,'no_training':True,'no_model_intervention':True,
            'original_files_preserved':True,'teacher_preserved':True,'frozen_student_preserved':True,'matched_windows':paired['windows'],
            'elapsed_seconds':time.monotonic()-started,'summary':{str(k):v['summary'] for k,v in results.items()},
            'runtime_warmups':{str(k):v['runtime_warmups'] for k,v in results.items()},
            'teacher_cache_all_bitwise':all(c['bitwise_equal'] for v in results.values() for c in v['teacher_cache_checks']),
            'saved_quality_checks_passed':True,'protected_files_sha256_after':after_hashes,'automatic_promotion':False})
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc)};base.write_json(args.out/'failure.json',failure);raise
    finally:
        base.write_json(args.out/'preservation.json',{'failure':failure,'teacher_preserved':resume.continuation.teacher_versions(teacher)==teacher_state,
            'frozen_student_preserved':screen.frozen_versions(model)==frozen,'original_files_preserved':all(base.sha(p)==v for p,v in protected.items())})

if __name__=='__main__':main()
