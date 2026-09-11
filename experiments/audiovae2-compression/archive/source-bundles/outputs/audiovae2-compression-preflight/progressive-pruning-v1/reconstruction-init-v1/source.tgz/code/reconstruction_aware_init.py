"""Calibration-only delta-ridge refits of existing384/256 native operators.

No new layer, gradient training, development fit, or checkpoint promotion.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import time

import torch
from torch import nn

import diagnose_progressive_silence as audit
import diagnose_channel_contributions_v2 as common

group,base,control,progressive = audit.group,audit.base,audit.control,audit.progressive
VERSION = 'audiovae2_reconstruction_aware_initialization_v1'
RIDGE, CHUNK_ROWS = 1e-6, 1024


def native_design(x,stride=5,start=0,stop=None):
    """Sparse phase-coded rows, preserving previous context before masking."""
    if x.ndim!=3 or x.shape[-1]<1 or not x.is_floating_point() or stride!=5:
        raise ValueError('Expected native stage3 stride5 floating input')
    stop=x.shape[-1]*stride if stop is None else stop
    if type(start)is not int or type(stop)is not int or not 0<=start<stop<=x.shape[-1]*stride:
        raise ValueError('Invalid native output row extent')
    index=torch.arange(start,stop,device=x.device);frame=index//stride;phase=index%stride
    current=x.index_select(-1,frame)
    previous=x.index_select(-1,(frame-1).clamp_min(0))*(frame>0).to(x)[None,None,:]
    pair=torch.cat((current,previous),1);cin=x.shape[1]
    result=x.new_zeros((x.shape[0],2*cin*stride,stop-start))
    for p in range(stride):result[:,p*2*cin:(p+1)*2*cin,phase==p]=pair[...,phase==p]
    return result


def native_affine_from_weight(w):
    if w.ndim!=3 or w.shape[-1]!=10: raise ValueError('Expected Cin,Cout,10 native weight')
    return torch.cat([w[:,:,tap].T for p in range(5) for tap in (p,p+5)],1)


def native_weight_from_affine(a,cin,stride=5):
    if stride!=5 or a.ndim!=2 or a.shape[1]!=2*cin*stride or not torch.isfinite(a).all():
        raise ValueError('Expected five native current/previous affine blocks')
    weight=a.new_empty((cin,a.shape[0],2*stride))
    for p in range(stride):
        weight[:,:,p]=a[:,p*2*cin:p*2*cin+cin].T
        weight[:,:,p+stride]=a[:,p*2*cin+cin:(p+1)*2*cin].T
    return weight


def residual_target(teacher_ru_output,current_student_skip,selected_outputs):
    teacher=teacher_ru_output.detach()[:,selected_outputs].double()
    skip=current_student_skip.detach().double()
    if teacher.shape!=skip.shape: raise ValueError('Selected complete RU and actual student skip shapes differ')
    return teacher-skip


def _validate_module(module):
    if module.bias is None: raise ValueError('The existing shared output bias is required')
    if isinstance(module,nn.ConvTranspose1d):
        if (module.stride!=(5,) or module.kernel_size!=(10,) or module.padding!=(0,)
                or module.output_padding!=(0,) or module.dilation!=(1,) or module.groups!=1):
            raise ValueError('Expected native causal stride5/kernel10 with trim5')
        return True
    if not isinstance(module,nn.Conv1d) or module.kernel_size!=(1,) or module.groups!=1 or module.stride!=(1,) or module.padding!=(0,):
        raise ValueError('Expected existing pointwise residual mixer')
    return False


def _target_energy(previous,delta,weights):
    d=delta.detach().movedim(1,-1).reshape(-1,delta.shape[1]).double()
    w=weights.reshape(-1).double();keep=w>0;d,w=d[keep],w[keep]
    if not len(w): return previous
    n=w.sum();mean=(d*w[:,None]).sum(0)/n;centered=((d-mean).square()*w[:,None]).sum()
    raw=(d.square()*w[:,None]).sum()
    if previous is None:return {'n':n,'mean':mean,'centered':centered,'raw':raw}
    total=previous['n']+n;shift=mean-previous['mean']
    previous['centered']+=centered+previous['n']*n/total*shift.square().sum()
    previous['mean']+=shift*(n/total);previous['n']=total;previous['raw']+=raw
    return previous


@torch.no_grad()
def fit_delta(module,observations,ridge=RIDGE,chunk_rows=CHUNK_ROWS):
    native=_validate_module(module)
    if ridge!=RIDGE or type(chunk_rows)is not int or chunk_rows<1:
        raise ValueError('Use the predeclared fixed delta ridge and positive chunk size')
    stats=energy=None;count=valid_rows=partial_rows=0;source_ids=[]
    for row in observations:
        x,target,current,weights=(row[k] for k in ('input','target','current_output','weights'))
        length=x.shape[-1]*5 if native else x.shape[-1]
        if (target.shape!=current.shape or target.shape!=(x.shape[0],module.out_channels,length)
                or weights.shape!=(x.shape[0],1,length) or (weights<0).any()
                or not torch.isfinite(weights).all()): raise ValueError('Fit observation geometry differs')
        if 'source_id' in row:source_ids.append(row['source_id'])
        delta=target.detach().double()-current.detach().double()
        for start in range(0,length,chunk_rows):
            stop=min(length,start+chunk_rows)
            design=native_design(x,start=start,stop=stop) if native else x[...,start:stop]
            w=weights[...,start:stop];d=delta[...,start:stop]
            stats=common.accumulate_affine(stats,design,d,w)
            energy=_target_energy(energy,d,w)
            valid_rows+=int((w>0).sum());partial_rows+=int(((w>0)&(w<(8 if native else 40))).sum())
        count+=1
    if stats is None or energy is None:raise ValueError('No valid calibration rows')
    if source_ids and len(source_ids)!=len(set(source_ids)):raise ValueError('Repeated calibration source within an operator fit')
    a,c,report=common.fit_affine(stats,ridge)
    covariance=stats['centered_xx']/stats['n'];covariance=(covariance+covariance.T)/2
    lam=report['ridge_lambda']
    if lam:
        _,info=torch.linalg.cholesky_ex(covariance+lam*torch.eye(a.shape[1],device=a.device,dtype=a.dtype))
        if int(info)!=0:raise RuntimeError('Regularized fit covariance failed positive Cholesky guard')
    centered_error=energy['centered']-2*(a*stats['centered_xd'].T).sum()+(a@stats['centered_xx']*a).sum()
    mean_error=stats['mean_d']-a@stats['mean_x']-c
    error=float(centered_error+stats['n']*mean_error.square().sum())
    floor=128*torch.finfo(torch.float64).eps*max(float(energy['raw']),1.)
    if not math.isfinite(error) or error < -floor:raise RuntimeError('Invalid fitted reconstruction energy')
    elements=float(stats['n'])*module.out_channels
    weight=native_weight_from_affine(a,module.in_channels) if native else a.unsqueeze(-1)
    report.update(observations=count,source_ids=source_ids,valid_feature_rows=valid_rows,partial_feature_rows=partial_rows,
        native_stride=5 if native else 1,native_kernel=10 if native else 1,
        native_trim=5 if native else 0,shared_bias=True,fit='delta from the current existing native operator',
        weighted_error_before=float(energy['raw'])/elements,weighted_error_after=max(0.,error)/elements,
        error_after_scope='Solver-predicted FP64 calibration error before FP32 WN writeback; held-out native inference is evaluated separately',
        raw_after_error_sum=error,small_negative_energy_roundoff_clamped=error<0,
        cholesky_positive=True if lam else None,chunk_rows=chunk_rows,
        target='Full teacher upsample output' if native else 'Selected complete teacher RU minus CURRENT student skip',
        samples_per_cell=8 if native else 40,all_valid_rows_included=True)
    return weight,c,report


@torch.no_grad()
def apply_delta(module,weight_delta,bias_delta):
    _validate_module(module);weight=group.effective_weight(module).detach()
    if weight_delta.shape!=weight.shape or bias_delta.shape!=module.bias.shape:
        raise ValueError('Correction must retain the original native operator shapes')
    desired=weight.double()+weight_delta.double();bias=module.bias.detach().double()+bias_delta.double()
    if not torch.isfinite(desired).all() or not torch.isfinite(bias).all():raise ValueError('Nonfinite native correction')
    group.assign_effective_weight(module,desired.to(weight),bias.to(module.bias))
    return {'weight':control.compare_tensors(group.effective_weight(module),desired.to(weight)),
            'bias':control.compare_tensors(module.bias,bias.to(module.bias))}


@torch.no_grad()
def sequential_fit(model,teacher,calibration,sites,observe_fn,on_report=None):
    """Apply each72-source fit, then recompute the actual chain for the next."""
    reports=[]
    for site in sites:
        before=control.state_hash(model.decoder);last=[]
        def observations():
            for crop in calibration:
                row=observe_fn(model,teacher,crop,site)
                last[:]=[row]
                yield row
        module=model.decoder.get_submodule(site.path)
        dw,db,report=fit_delta(module,observations())
        representative=last[0];x=representative['input']
        predicted=representative['current_output'].double()+common.linear_response(module,x.double(),dw,False)+db[None,:,None]
        desired=(group.effective_weight(module).detach().double()+dw).to(x)
        report['effective_writeback']=apply_delta(module,dw,db)
        actual=module(x)
        desired_native=common.linear_response(module,x,desired,True)
        report['last_source_native_fold_parity']=control.compare_tensors(actual,desired_native)
        report['analytic_double_correction_parity']=control.compare_tensors(actual,predicted.to(actual))
        report.update(site=site.name,path=site.path,input_candidate_state_sha256=before,
                      output_candidate_state_sha256=control.state_hash(model.decoder),
                      inputs_recomputed_after_previous_fit=True)
        reports.append(report)
        if on_report is not None:on_report(reports)
        if (not report['last_source_native_fold_parity']['allclose_existing']
                or not all(r['allclose_existing'] for r in report['effective_writeback'].values())):
            raise RuntimeError('Fitted native operation or effective weights differ from the intended FP32 operator')
    return reports


@torch.no_grad()
def observe_operation(model,teacher,crop,site):
    z,target,valid,_=base.batch([crop])
    with audit.capture_sites(teacher.model.decoder,[site]) as tc:
        tr=base.teacher_forward(teacher,z)
    common.warm_student(model,tr['group_input'])
    with audit.capture_sites(model.decoder,[site]) as sc:
        model.group_from_input(tr['group_input'])
    t,s=tc[site.name],sc[site.name]
    desired=(t['output'].detach().double() if site.stride>1
             else residual_target(t['ru_output'],s['skip_input'],site.kept_outputs))
    return {'source_id':crop['source_id'],'input':s['input'],'target':desired,
            'current_output':s['output'],'weights':common.cell_weights(valid,s['output'].shape[-1])}


def calibration_coverage(crops):
    quiet=[];valid_samples=0;starting_sources=[]
    for crop in crops:
        _,target,_,spans=base.batch([crop]);a,b=spans[0];t=target[...,a:b]
        valid_samples+=b-a
        if crop['start_frame']==0 and crop['context_frames']==0:starting_sources.append(crop['source_id'])
        raw=common.diagnostic_quiet_metrics(t,t,torch.ones_like(t,dtype=torch.bool))
        quiet.extend(control.joint.quiet_audit.describe_quiet_windows(t,t,crop,raw))
    near=[r for r in quiet if r['teacher_rms']<=1e-5]
    startup=[r for r in near if r['source_start_sample']<960]
    return {'sources':len(crops),'all_valid_samples':valid_samples,'duration_seconds':valid_samples/48000,
            'source_starting_crop_count':len(starting_sources),'source_starting_crop_ids':starting_sources,
            'teacher_quiet_windows':len(quiet),'teacher_quiet_samples':sum(r['valid_samples'] for r in quiet),
            'teacher_near_windows':len(near),'teacher_near_samples':sum(r['valid_samples'] for r in near),
            'teacher_near_startup_windows':len(startup),'teacher_near_startup_samples':sum(r['valid_samples'] for r in startup),
            'teacher_near_startup_source_ids':[r['source_id'] for r in startup],
            'teacher_near_startup_reference_zero':sum(r['source_reference_exact_zero'] for r in startup),
            'fit_sampling':'All valid feature cells, weighted by actual valid48k samples per cell; no subsampling or extra startup weight'}


@torch.no_grad()
def evaluate_candidate(model,teacher,crops,objective,metadata,out,label):
    from unified_monitor import UnifiedMonitor
    from joint_recovery_gates_v2 import summarize_regions
    def capture(rows):
        control._write_rows(out/(label+'-quiet-windows.jsonl.gz'),rows)
        return summarize_regions(rows)
    report=control.joint.evaluate_review(UnifiedMonitor(None,{},metadata),model,teacher,crops,objective,summarize=capture)
    base.write_json(out/(label+'-development.json'),report)
    return report


def _assert_only_sites_changed(before,model,sites):
    after=audit._versions(model.decoder);allowed=tuple(site.path+'.' for site in sites)
    if len(before)!=len(after):raise RuntimeError('Calibration changed native model parameter topology')
    for a,b in zip(before,after):
        if a[:3]!=b[:3] or (a!=b and not a[0].startswith(allowed)):
            raise RuntimeError('Calibration changed an unselected or frozen model tensor')


@torch.no_grad()
def main():
    from compare_accumulation_v1 import numeric_reference_check
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('step0','checkpoint','manifest','assets','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Use a new isolated calibration output directory')
    if any(args.out.resolve().is_relative_to(p.resolve()) for p in (args.step0.parent,args.checkpoint.parent)):
        raise ValueError('Calibration output must not overlap preserved training runs')
    zero,trained,authenticated=audit.checkpoint_inputs(args)
    if base.sha(args.manifest) not in authenticated['launch']['protected'].values():
        raise ValueError('Use the original authenticated calibration/development manifest')
    process=control.gpu_idle_snapshot()
    manifest,pools,_=base.load_data(args.manifest)
    calibration,development=pools['calibration'],pools['development']
    if len(calibration)!=72 or len(development)!=96:raise ValueError('Expected fixed72 calibration and96 development sources')
    split=common.validate_fit_sources(calibration,development)
    paths=[args.step0,args.checkpoint,args.manifest,args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',
           Path(manifest['cache_path']),Path(manifest['overlay_receipt_path']),args.step0.parent/'development-step0.json',
           Path(__file__),Path(audit.__file__),Path(common.__file__),Path(control.__file__),Path(group.__file__),
           Path(progressive.__file__),Path(base.__file__)]
    protected={**authenticated['protected'],**{str(p.resolve()):base.sha(p) for p in paths}}
    base.policy()
    if str(torch.__version__)!=authenticated['launch']['torch'] or torch.backends.cudnn.version()!=authenticated['launch']['cudnn']:
        raise ValueError('Use the preserved FP32 runtime and backend')
    coverage=calibration_coverage(calibration)
    args.out.mkdir(parents=True)
    base.write_json(args.out/'launch.json',{'version':VERSION,'protected':protected,**split,
        'selection':zero['selection'],'original_step0_sha256':audit.STEP0_SHA,'preserved_step5000_sha256':audit.STEP5000_SHA,
        'ridge_factor':RIDGE,'chunk_rows':CHUNK_ROWS,'covariance_dtype':'float64','inference_dtype':'float32',
        'ridge_policy':'Delta ridge lambda=1e-6*trace(centered covariance)/design_dimension, unpenalized one shared bias',
        'variants':{'A':['stage3_up'],'B':['stage2_ru1','stage2_ru2','stage2_ru3','stage3_up']},
        'calibration_coverage':coverage,'process_snapshot':process,'backend':common.replay.backend_state(),
        'neural_training_updates':0,'optimizer_created':False,'added_inference_modules':0,'automatic_promotion':False})
    base.write_json(args.out/'calibration-coverage.json',coverage)
    teacher=initial=None;states={};results={};fits={};failure=None;started=time.monotonic()
    try:
        teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
        initial=progressive.initialize_from_teacher(teacher.model.decoder,zero['selection'])
        initial.load_group_state_dict(zero['group']);initial.eval()
        pristine=audit.validate_pristine(teacher.model.decoder,initial,zero['selection'])
        states={'teacher':control.state_hash(teacher.model),'original_step0':control.state_hash(initial.decoder)}
        sites=audit.four_sites(teacher.model.decoder,zero['selection'])
        objective=base.objective();metadata={r['source_id']:r for r in manifest['splits']['development']['rows']}
        audit.warm_models(teacher,[initial],development)
        results['baseline']=evaluate_candidate(initial,teacher,development,objective,metadata,args.out,'baseline-step0')
        saved=json.loads((args.step0.parent/'development-step0.json').read_text())
        parity=numeric_reference_check(results['baseline'],saved)
        base.write_json(args.out/'baseline-parity.json',parity)
        if not parity['passed']:raise RuntimeError('Original step0 development baseline did not reproduce')
        base.event('reconstruction_baseline_verified',development_sources=96)
        for variant,chosen in (('A',[sites[-1]]),('B',sites)):
            model=progressive.initialize_from_teacher(teacher.model.decoder,zero['selection'])
            model.load_group_state_dict(zero['group']);model.eval()
            if control.state_hash(model.decoder)!=states['original_step0']:
                raise RuntimeError('Each variant must independently start from the exact original step0')
            before=audit._versions(model.decoder);observed={}
            def observe(current,t,crop,site):
                row=observe_operation(current,t,crop,site)
                observed[site.name]=observed.get(site.name,0)+1
                if observed[site.name]%12==0:
                    base.event('reconstruction_calibration_rows',variant=variant,site=site.name,sources=observed[site.name],expected=72)
                return row
            def on_report(reports):
                base.write_json(args.out/f'variant-{variant}-fits.json',reports)
                r=reports[-1]
                base.event('reconstruction_operator_fitted',variant=variant,site=r['site'],
                    solver_predicted_error_before=r['weighted_error_before'],solver_predicted_error_after=r['weighted_error_after'])
            fits[variant]=sequential_fit(model,teacher,calibration,chosen,observe,on_report)
            _assert_only_sites_changed(before,model,chosen)
            state={site.path:{k:v.detach().cpu().clone() for k,v in model.decoder.get_submodule(site.path).state_dict().items()}
                   for site in chosen}
            artifact=args.out/f'variant-{variant}-native-operators.pt'
            torch.save({'format':VERSION,'variant':variant,'base_step0_sha256':audit.STEP0_SHA,'selection':zero['selection'],
                'fit_source_ids':split['fit_source_ids'],'ridge':RIDGE,'operators':state},artifact)
            candidate_hash=control.state_hash(model.decoder)
            audit.warm_models(teacher,[model],development)
            results[variant]=evaluate_candidate(model,teacher,development,objective,metadata,args.out,'variant-'+variant)
            if control.state_hash(model.decoder)!=candidate_hash:raise RuntimeError('Evaluation changed fitted native candidate')
            _assert_only_sites_changed(before,model,chosen)
            base.write_json(args.out/f'variant-{variant}-receipt.json',{'operators_sha256':base.sha(artifact),
                'candidate_state_sha256':candidate_hash,'unchanged_frozen_and_unselected_tensors':True,
                'changed_native_paths':[s.path for s in chosen],'full_group_widths':[384,256,128],
                'all9_residual_units_preserved':True,'original_step0_pristine':audit._public_pristine(pristine),
                'neural_training_updates':0,'extra_inference_modules':0,'automatic_promotion':False})
            base.event('reconstruction_variant_evaluated',variant=variant,development_sources=96)
            del state,model
    except BaseException as exc:
        failure=repr(exc);raise
    finally:
        files=all(base.sha(p)==s for p,s in protected.items())
        unchanged={'teacher':teacher is None or not states or control.state_hash(teacher.model)==states['teacher'],
                   'original_step0':initial is None or not states or control.state_hash(initial.decoder)==states['original_step0']}
        complete=failure is None and set(results)=={'baseline','A','B'} and files and all(unchanged.values())
        summary={'version':VERSION,'complete':complete,'failure':failure,'files_preserved':files,'states_preserved':unchanged,
            'calibration_sources':72,'development_sources':96,'calibration_coverage':coverage,'neural_training_updates':0,
            'automatic_promotion':False,'elapsed_seconds':time.monotonic()-started,
            'results':{name:{'aggregate':r['aggregate'],'quiet_regions':r['quiet_regions'],
                              'overview_window_metrics':r['overview_window_metrics'],'recovery_window_metrics':r['recovery_window_metrics']}
                       for name,r in results.items()}}
        base.write_json(args.out/'completed.json',summary)
        if not files or not all(unchanged.values()):raise RuntimeError('Calibration altered preserved original assets/state')


if __name__=='__main__':main()
