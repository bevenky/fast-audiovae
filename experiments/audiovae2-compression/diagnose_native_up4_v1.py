"""Bounded native-up4 regression and downstream-coordinate diagnostics.

No optimizer or neural training. Original initializer and retained trained4625
are evaluated independently. Teacher-feature injections are diagnostic oracles.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
import torch
import native_upsampler_fit_v1 as fit
import compare_projected_hints_v1 as retained
from unified_monitor import UnifiedMonitor

common=fit.common;base=common.base;screen=common.screen;resume=common.resume;replay=common.replay;group=common.group
VERSION='audiovae2_native_up4_reconstruction_v1'
SPEC=common.Operation('up4',fit.PATH,[],[],12000,2)


def regional_errors(prediction,target,masks):
    rows={}
    for region,mask in masks.items():
        weights=common.cell_weights(mask,target.shape[-1])
        rows[region]={}
        for label,selection in [('all',slice(None)),('phase0',slice(0,None,2)),('phase1',slice(1,None,2))]:
            w=weights[...,selection].double();t=target[...,selection].double();p=prediction[...,selection].double()
            count=float(w.sum())*t.shape[1];keep=w>0
            error=(p-t).masked_fill(~keep,0);t=t.masked_fill(~keep,0)
            rows[region][label]={'weighted_elements':count,'squared_error_sum':float((error.square()*w).sum()),
                'absolute_error_sum':float((error.abs()*w).sum()),'teacher_square_sum':float((t.square()*w).sum()),
                'max_abs':float(error.abs().max()) if count else None,
                'error':common.weighted_tensor_summary(error,w)}
    return rows


def pool_regions(rows):
    result={}
    for row in rows:
        for region,phases in row['regions'].items():
            for phase,values in phases.items():
                pooled=result.setdefault(region,{}).setdefault(phase,{k:0. for k in ('weighted_elements','squared_error_sum','absolute_error_sum','teacher_square_sum')})
                for key in pooled:pooled[key]+=values[key]
    for phases in result.values():
        for values in phases.values():
            n=values['weighted_elements'];values['mse']=values['squared_error_sum']/n if n else None
            values['mae']=values['absolute_error_sum']/n if n else None
            values['relative_squared_error']=values['squared_error_sum']/values['teacher_square_sum'] if values['teacher_square_sum'] else None
    return result


def check_cache(trace,crop,target,valid):
    actual=trace['waveform'][valid];expected=target[valid]
    result={'source_id':crop['source_id'],'bitwise_equal':torch.equal(actual,expected),
        'max_abs':float((actual-expected).abs().max()),'passed':torch.allclose(actual,expected,atol=1e-5,rtol=1e-4)}
    if not result['passed']:raise RuntimeError('Pristine teacher differs from the fixed cached target')
    return result


@torch.no_grad()
def fit_model(model,teacher,crops,selection,*,teacher_selected=False):
    stats=None;checks=[]
    for crop in crops:
        z,target,valid,_=base.batch([crop])
        with common.capture_operations(teacher.model.decoder,[SPEC]) as tc:trace=base.teacher_forward(teacher,z)
        checks.append(check_cache(trace,crop,target,valid))
        if teacher_selected:x=tc['up4']['input'][:,selection['stage3_indices'],:]
        else:
            common.warm_student(model,trace['group_input'])
            with common.capture_operations(model.decoder,[SPEC]) as sc:model.group_from_input(trace['group_input'])
            x=sc['up4']['input']
        stats=fit.accumulate_native(stats,x,tc['up4']['output'],valid)
    w,b,report=fit.fit_native(stats,len(selection['stage3_indices']),ridge=1e-6)
    report.update(source_ids=[c['source_id'] for c in crops],teacher_target_checks=checks,
        input_scope='original teacher retained inputs' if teacher_selected else 'actual current student upstream input',
        target_scope='full original teacher up4 output, before all three stage4 residual units')
    return w,b,report


@torch.no_grad()
def check_native_fold(model,teacher,crop,w,b):
    z,_,_,_=base.batch([crop]);trace=base.teacher_forward(teacher,z)
    common.warm_student(model,trace['group_input'])
    with common.capture_operations(model.decoder,[SPEC]) as sc:model.group_from_input(trace['group_input'])
    x=sc['up4']['input'];module=model.decoder.get_submodule(fit.PATH)
    expected=common.affine_response(fit.native_affine_from_weight(w),b,fit.native_design(x))
    with fit.temporary_native_fit(module,w,b):actual=module(x)
    error=actual.double()-expected.double()
    report={'source_id':crop['source_id'],'input_frames':x.shape[-1],'output_frames':actual.shape[-1],
        'max_abs':float(error.abs().max()),'residual_rms':float(error.square().mean().sqrt()),
        'passed':torch.allclose(actual,expected,atol=1e-5,rtol=1e-4),'atol':1e-5,'rtol':1e-4,
        'bitwise_equal':torch.equal(actual,expected)}
    if actual.shape[-1]!=2*x.shape[-1] or not report['passed']:raise RuntimeError('Native weight folding or causal trim parity failed')
    return report


@torch.no_grad()
def score_local(model,teacher,crops,w,b,selection,*,teacher_selected=False):
    rows=[];checks=[];module=model.decoder.get_submodule(fit.PATH)
    for crop in crops:
        z,target,valid,_=base.batch([crop])
        with common.capture_operations(teacher.model.decoder,[SPEC]) as tc:trace=base.teacher_forward(teacher,z)
        checks.append(check_cache(trace,crop,target,valid))
        if teacher_selected:x=tc['up4']['input'][:,selection['stage3_indices'],:]
        else:
            common.warm_student(model,trace['group_input'])
            with common.capture_operations(model.decoder,[SPEC]) as sc:model.group_from_input(trace['group_input'])
            x=sc['up4']['input']
        # Native arithmetic with one shared bias; no state mutation for this local score.
        prediction=torch.nn.functional.conv_transpose1d(x,w.to(x),b.to(x),stride=2)[...,:2*x.shape[-1]]
        rows.append({'source_id':crop['source_id'],'regions':regional_errors(prediction,tc['up4']['output'],common.region_masks(target,valid,crop))})
    return {'aggregate':pool_regions(rows),'rows':rows,'teacher_target_checks':checks}


class MeasurementView:
    def __init__(self,model,teacher,crops,variant,save_trace):
        self.model,self.teacher,self.crops,self.variant,self.save_trace=model,teacher,crops,variant,save_trace
        self.cursor=0;self.rows=[];self.waveform_rows=[];self.teacher_capture=None;self.teacher_trace=None
    def group_from_input(self,x):
        crop=self.crops[self.cursor];_,target,valid,_=base.batch([crop]);self.current=crop,target,valid
        common.warm_student(self.model,x)
        if self.variant=='teacher_group_end':return self.teacher_trace['group_output'].clone()
        with ExitStack() as stack:
            if self.variant=='teacher_up4':
                stack.enter_context(common.temporary_output_change(self.model.decoder.get_submodule(fit.PATH),lambda _out:self.teacher_capture['up4']['output'].clone()))
            with common.capture_operations(self.model.decoder,[SPEC]) as sc:h=self.model.group_from_input(x)
        self.rows.append({'source_id':crop['source_id'],'regions':regional_errors(sc['up4']['output'],self.teacher_capture['up4']['output'],common.region_masks(target,valid,crop))})
        return h
    def suffix_from_group(self,h):
        prediction=self.model.suffix_from_group(h);crop,target,valid=self.current
        regions={}
        for label,mask in common.region_masks(target,valid,crop).items():
            p,t=prediction[mask].double(),target[mask].double();energy=float(t.square().sum());dot=float((p*t).sum());pe=float(p.square().sum())
            regions[label]={'samples':p.numel(),'teacher_square_sum':energy,'prediction_square_sum':pe,'dot_sum':dot,
                'gain':dot/energy if energy else None,'rms_ratio':(pe/energy)**.5 if energy else None,
                'squared_error_sum':float((p-t).square().sum()),'absolute_error_sum':float((p-t).abs().sum())}
        self.waveform_rows.append({'source_id':crop['source_id'],'regions':regions})
        if self.variant=='teacher_group_end' and not torch.allclose(prediction[valid],target[valid],atol=1e-5,rtol=1e-4):
            raise RuntimeError('Teacher group-end into unchanged suffix does not recover the teacher waveform')
        if self.save_trace:self.save_trace(crop,target,prediction,valid)
        self.cursor+=1;return prediction


@torch.no_grad()
def evaluate_variant(model,teacher,crops,spectral,variant,save_trace=None):
    view=MeasurementView(model,teacher,crops,variant,save_trace);original=base.teacher_forward;checks=[]
    def observed(t,z):
        with common.capture_operations(t.model.decoder,[SPEC]) as captured:trace=original(t,z)
        crop=crops[len(checks)];_,target,valid,_=base.batch([crop]);checks.append(check_cache(trace,crop,target,valid))
        view.teacher_capture=captured;view.teacher_trace=trace;return trace
    base.teacher_forward=observed
    try:quality=UnifiedMonitor(None,{},{}).evaluate(view,teacher,crops,spectral)
    finally:base.teacher_forward=original
    if view.cursor!=len(crops) or len(checks)!=len(crops):raise RuntimeError('Incomplete held-out source accounting')
    waveform_aggregate={}
    for row in view.waveform_rows:
        for region,values in row['regions'].items():
            aggregate=waveform_aggregate.setdefault(region,{k:0. for k in ('samples','teacher_square_sum','prediction_square_sum','dot_sum','squared_error_sum','absolute_error_sum')})
            for key in aggregate:aggregate[key]+=values[key]
    for v in waveform_aggregate.values():
        e=v['teacher_square_sum'];n=v['samples'];v['gain']=v['dot_sum']/e if e else None
        v['rms_ratio']=(v['prediction_square_sum']/e)**.5 if e else None
        v['mae']=v['absolute_error_sum']/n if n else None;v['mse']=v['squared_error_sum']/n if n else None
    return {'quality':quality,'up4_local':{'aggregate':pool_regions(view.rows),'rows':view.rows},
        'waveform_regions':{'aggregate':waveform_aggregate,'rows':view.waveform_rows},'teacher_target_checks':checks,
        'variant':variant,'no_neural_training':True}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    for name in ('checkpoint','candidate','anchor-checkpoint','screen-out','base-out','manifest','fresh-manifest','shards','assets','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a new isolated diagnostic directory')
    base.policy()
    _,selection,_,manifest,pools,original,_,original_sha,_=resume.authenticate(args)
    metadata,selection2,initial=screen.authenticate_preflight(args.base_out)
    if selection2!=selection:raise ValueError('Initial and trained channel selection differ')
    data=replay.FreshTrainingData(args.fresh_manifest,args.manifest,pools,args.shards)
    ids=list(data.source_ids[10500:12000]);candidate,reference,candidate_sha=retained.authenticate_candidate(args.candidate,original,ids)
    if (candidate['identity']['start_checkpoint_sha256']!=original_sha
            or candidate['identity']['runner_sha256']!=base.sha(retained.previous.__file__)
            or candidate['identity']['resume_identity']!=original['resume_identity']
            or candidate['identity']['source_ids']!=ids):raise ValueError('Retained candidate anchor/code/source identity changed')
    calibration,development=pools['calibration'],pools['development'];split=common.validate_fit_sources(calibration,development)
    if len(calibration)!=72 or len(development)!=96:raise ValueError('Fixed fitting/development panels changed')
    protected_paths=[args.checkpoint,args.candidate,args.anchor_checkpoint,args.manifest,args.fresh_manifest,
        args.base_out/'initial.pt',args.base_out/'preflight.json',args.base_out/'channel-selection.json',
        args.candidate.parent/'completed.json',args.candidate.parent.parent/'completed.json',args.candidate.parent/'development-source1500.json',
        args.checkpoint.parent/'checkpoint-step5000.pt',args.checkpoint.parent/'checkpoint-step5000.json',
        args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',Path(__file__),Path(fit.__file__),Path(common.__file__),Path(group.__file__),Path(base.__file__)]
    protected={str(p):base.sha(p) for p in protected_paths}
    teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    teacher_state=resume.continuation.teacher_versions(teacher)
    models={};original_states={};frozen={}
    for name,state in [('sliced_initial',initial['group']),('trained4625',candidate['group'])]:
        model=base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices'])
        model.load_group_state_dict(state);models[name]=model
        original_states[name]={k:v.detach().cpu().clone() for k,v in model.group_state_dict().items()};frozen[name]=screen.frozen_versions(model)
    args.out.mkdir();started=time.monotonic();failure=None;results={};traces={};trace_index=[]
    selected_ids=list(replay.CASE_IDS)
    identity={'version':VERSION,'helper_sha256':base.sha(fit.__file__),'runner_sha256':base.sha(__file__),
        'candidate_sha256':candidate_sha,'initial_sha256':metadata['initial_sha256'],**split,'selection':selection,
        'no_neural_training':True,'automatic_promotion':False,'protected':protected,'backend':replay.backend_state(),
        'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'ridge_factor':1e-6,
        'fit_operator':'Native stride2/kernel4, current/previous input, shared output-channel bias; 4 valid waveform samples per output cell',
        'teacher_selected_fit':'Local reconstructibility only; never installed in an adapted model'}
    base.write_json(args.out/'launch.json',identity)
    def save_trace_for(name):
        def save(crop,target,prediction,valid):
            if crop['source_id'] not in selected_ids:return
            index=selected_ids.index(crop['source_id'])
            for label,(a,b) in common.trace_intervals(target,valid,crop).items():
                key=f'source{index}_{label}'
                if key+'_teacher' not in traces:
                    traces[key+'_teacher']=target[0,0,a:b].cpu().numpy().copy()
                    trace_index.append({'key':key,'source_id':crop['source_id'],'interval':label,'crop_samples':[a,b],
                        'absolute_source_samples':[crop['context_start_frame']*1920+a,crop['context_start_frame']*1920+b],'sample_rate':48000})
                traces[key+'_'+name]=prediction[0,0,a:b].cpu().numpy().copy()
        return save
    try:
        spectral=base.objective()
        for name,model in models.items():
            w,b,fit_report=fit_model(model,teacher,calibration,selection)
            fit_report['native_fold_parity']=check_native_fold(model,teacher,calibration[0],w,b)
            base.write_json(args.out/(name+'-fit.json'),fit_report)
            torch.save({'format':VERSION,'weight':w.cpu(),'bias':b.cpu(),'base':name,'identity':identity},args.out/(name+'-native-fit.pt'))
            fit_local=score_local(model,teacher,calibration,w,b,selection)
            base.write_json(args.out/(name+'-fit-local.json'),fit_local)
            for variant in ('baseline','native_fit','teacher_up4')+ (('native_fit_original_residuals',) if name=='trained4625' else ()):
                with ExitStack() as stack:
                    if variant.startswith('native_fit'):stack.enter_context(fit.temporary_native_fit(model.decoder.get_submodule(fit.PATH),w,b))
                    if variant=='native_fit_original_residuals':stack.enter_context(fit.temporary_original_residuals(model.decoder,teacher.model.decoder))
                    report=evaluate_variant(model,teacher,development,spectral,variant,save_trace_for(name+'_'+variant))
                if not replay.compare_tree(dict(model.group_state_dict()),original_states[name])['equal']:raise RuntimeError('Scoped intervention did not restore original group')
                if name=='trained4625' and variant=='baseline':
                    report['saved_quality_check']=retained.previous.numeric_reference_check(report['quality'],reference)
                    if not report['saved_quality_check']['passed']:raise RuntimeError('Retained trained baseline quality changed')
                key=name+'-'+variant;base.write_json(args.out/(key+'.json'),report)
                results[key]=report['quality']['aggregate'];base.event('native_up4_progress',completed=key,seconds=time.monotonic()-started)
        model=models['sliced_initial'];w,b,selected_fit=fit_model(model,teacher,calibration,selection,teacher_selected=True)
        base.write_json(args.out/'teacher-retained-input-fit.json',selected_fit)
        for label,crops in [('fit',calibration),('development',development)]:
            report=score_local(model,teacher,crops,w,b,selection,teacher_selected=True)
            base.write_json(args.out/('teacher-retained-input-'+label+'.json'),report)
        control=evaluate_variant(model,teacher,development,spectral,'teacher_group_end',save_trace_for('teacher_group_end'))
        base.write_json(args.out/'teacher-group-end-control.json',control);results['teacher_group_end']=control['quality']['aggregate']
        # Completion is emitted only after byte/value checks of all protected state.
        preserved=all(replay.compare_tree(dict(m.group_state_dict()),original_states[name])['equal'] and screen.frozen_versions(m)==frozen[name] for name,m in models.items())
        teacher_ok=resume.continuation.teacher_versions(teacher)==teacher_state;files_ok=all(base.sha(p)==v for p,v in protected.items())
        if not preserved or not teacher_ok or not files_ok:raise RuntimeError('Protected state changed')
        base.write_json(args.out/'completed.json',{'version':VERSION,'completed':True,'variants':results,'elapsed_seconds':time.monotonic()-started,
            'original_groups_preserved':True,'teacher_preserved':True,'original_files_preserved':True,'no_neural_training':True,'automatic_promotion':False,
            'limits':['Native affine failure is not a proof of insufficient nonlinear capacity.',
                'Teacher-up4 injection is an oracle; adapted downstream residuals need not retain original teacher internal coordinates.',
                'Teacher-retained input fit is local-only and does not include actual upstream drift.',
                'Unchanged native operation shapes imply no added inference operations, but no CPU timing was measured.']})
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc)};base.write_json(args.out/'failure.json',failure);raise
    finally:
        import numpy as np
        if traces:np.savez_compressed(args.out/'waveform-snippets.npz',**traces);base.write_json(args.out/'waveform-snippets.json',trace_index)
        base.write_json(args.out/'preservation.json',{'failure':failure,'teacher_preserved':resume.continuation.teacher_versions(teacher)==teacher_state,
            'original_groups_preserved':{n:replay.compare_tree(dict(m.group_state_dict()),original_states[n])['equal'] for n,m in models.items()},
            'frozen_models_preserved':{n:screen.frozen_versions(m)==frozen[n] for n,m in models.items()},
            'original_files_preserved':all(base.sha(p)==v for p,v in protected.items())})

if __name__=='__main__':main()
