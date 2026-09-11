"""Read-only singleton teacher/student layer localization, no optimizer or updates."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import torch
import settings_screen as screen
from group_model import teacher_trace,effective_weight,_legacy_weight_hook
from audiovae_student.quiet_audio import quiet_window_metrics
base=screen.base
VERSION='audiovae2_regressing_layer_diagnosis_v1'
SOURCES=('es_419:train:13461728374156750135.wav','kn_in:train:15096009457771558023.wav',
         '2051-153962-0091','lb_lu:train:10437720223409828494.wav','freesound:428921','freesound:402835')
RATES={1:200,2:1200,3:6000,4:12000,5:24000,6:48000}

def capture(decoder,z):
    values={};handles=[]
    def post(name):
        def hook(module,args,result):values[name]=result.detach()
        return hook
    def residual(name):
        def hook(module,args,result):
            values[name+'.output']=result.detach()
            values[name+'.branch']=result.detach()-args[0].detach()
            values[name+'.input']=args[0].detach()
        return hook
    try:
        for stage in range(1,7):
            index=stage+1;block=decoder.model[index]
            handles.append(block.register_forward_hook(post(f'stage{stage}.output')))
            handles.append(block.block[0].register_forward_hook(post(f'stage{stage}.input_snake')))
            handles.append(block.block[1].register_forward_hook(post(f'stage{stage}.upsample')))
            if decoder.sr_cond_model[index] is not None:
                handles.append(decoder.sr_cond_model[index].register_forward_hook(post(f'stage{stage}.conditioned_input')))
            for i,unit in enumerate(block.block[2:],1):
                handles.append(unit.register_forward_hook(residual(f'stage{stage}.ru{i}')))
        for index,name in ((8,'head.snake'),(9,'head.pre_tanh'),(10,'head.waveform')):
            handles.append(decoder.model[index].register_forward_hook(post(name)))
        trace=teacher_trace(decoder,z)
        if not torch.equal(values['head.waveform'],trace['waveform']):raise RuntimeError('Hook output drift')
        return trace,values
    finally:
        for handle in handles:handle.remove()

def masks(target,valid):
    windows=quiet_window_metrics(target,target,valid)['windows'];quiet=torch.zeros_like(valid);near=torch.zeros_like(valid)
    for w in windows:
        if w['is_quiet']:
            sl=slice(w['start_sample'],w['stop_sample']);quiet[...,sl]=valid[...,sl]
            if w['teacher_rms']<=1e-5:near[...,sl]=valid[...,sl]
    return {'all':valid,'near_silence':near,'other_quiet':quiet&~near,'quiet':quiet,'active':valid&~quiet}

def stats(p,t,mask):
    if p.shape!=t.shape or mask.shape[-1]%p.shape[-1]:raise ValueError('Shape/rate mismatch')
    factor=mask.shape[-1]//p.shape[-1]
    w=mask.reshape(1,1,p.shape[-1],factor).sum(-1).double();n=float(w.sum())
    if not n:return {'valid_audio_samples':0}
    p,t=p.double(),t.double();e=p-t;channels=p.shape[1];count=n*channels
    def energy(x):return float((x.square()*w).sum())/count
    ep,et,ee=energy(p),energy(t),energy(e)
    means=(e*w).sum(-1)/n;dc=float(means.square().mean());dot=float((p*t*w).sum())/count
    return {'valid_audio_samples':int(n),'channels':channels,'rate_hz':48000//factor,
        'mae':float((e.abs()*w).sum())/count,'residual_rms':math.sqrt(ee),
        'teacher_rms':math.sqrt(et),'student_rms':math.sqrt(ep),'dc_error_rms':math.sqrt(dc),
        'ac_error_rms':math.sqrt(max(0.,ee-dc)),'signed_mean_error':float(means.mean()),
        'dc_error_energy_fraction':dc/ee if ee else 0.,
        'cosine':dot/math.sqrt(ep*et) if ep and et else None,
        'max_abs_error':float(e.abs().masked_select((w>0).expand_as(e)).max()),
        'gain_only_when_teacher_rms_gt_1e-5':math.sqrt(ep/et) if et>1e-10 else None}

def wave_windows(p,t,valid):
    report=quiet_window_metrics(p,t,valid);rows=[]
    for row in report['windows']:
        if not row['is_quiet']:continue
        a,b=row['start_sample'],row['stop_sample'];m=valid[...,a:b];e=(p[...,a:b]-t[...,a:b]).double()[m]
        dc=float(e.mean()) if e.numel() else 0.;ee=float(e.square().sum())
        rows.append({**row,'signed_dc_error':dc,'dc_energy_fraction':dc*dc*e.numel()/ee if ee else 0.,
                     'ac_error_rms':math.sqrt(max(0.,float(e.square().mean())-dc*dc))})
    near=[r for r in rows if r['teacher_rms']<=1e-5]
    return {'quiet_failed':report['quiet_failed_count'],'quiet_windows':len(rows),
            'near_silence_failed':sum(r['passed'] is False for r in near),'near_silence_windows':len(near),
            'windows':rows}

def layer_selection(name,selection):
    if name.startswith('head'):return None
    stage=int(name.split('.')[0][5:])
    if name.endswith('input_snake') or name.endswith('conditioned_input'):stage-=1
    return selection.get('stage2_indices' if stage==2 else 'stage3_indices' if stage==3 else '')

def layer_report(p,t,region_masks,selection):
    result={}
    for name in p:
        ref=t[name];original=ref.shape[1];indices=layer_selection(name,selection)
        if indices is not None:ref=ref.index_select(1,torch.tensor(indices,device=ref.device))
        if ref.shape!=p[name].shape:raise RuntimeError('Layer alignment differs: '+name)
        result[name]={'teacher_channels':original,'compared_channels':ref.shape[1],
            'scope':'selected_internal_coordinates_only' if indices is not None else 'shared_full_coordinates',
            'regions':{region:stats(p[name],ref,mask) for region,mask in region_masks.items()}}
    return result

def parameter_audit(decoder):
    result={'snakes':{},'convolutions':{},'finite':True}
    for name,module in decoder.named_modules():
        if hasattr(module,'alpha'):
            a=module.alpha.detach();result['snakes'][name]={'min':float(a.min()),'max':float(a.max()),
                 'minimum_abs':float(a.abs().min()),'finite':bool(torch.isfinite(a).all())}
        hook=_legacy_weight_hook(module)
        if hook:
            w=effective_weight(module).detach();v=module.weight_v.detach();g=module.weight_g.detach()
            finite=bool(torch.isfinite(w).all() and torch.isfinite(v).all() and torch.isfinite(g).all())
            result['finite'] &= finite
            result['convolutions'][name]={'weight_shape':list(w.shape),'wn_dim':hook.dim,'finite':finite,
                'weight_rms':float(w.square().mean().sqrt()),'min_abs_scale':float(g.abs().min()),
                'stride':list(module.stride),'dilation':list(module.dilation),'kernel_size':list(module.kernel_size),
                'groups':module.groups,'causal_padding':getattr(module,'_CausalConv1d__padding',None),
                'causal_transpose_padding':getattr(module,'_CausalTransposeConv1d__padding',None)}
    return result

def target_comparison(trace,target,valid,regions):
    return {'exact':bool(torch.equal(trace['waveform'][valid],target[valid])),
            'allclose_atol1e-5_rtol1e-4':bool(torch.allclose(trace['waveform'][valid],target[valid],atol=1e-5,rtol=1e-4)),
            'regions':{name:stats(trace['waveform'],target,mask) for name,mask in regions.items()}}

def main():
    ap=argparse.ArgumentParser()
    for key in ('root','assets','out'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--fresh-panel',type=Path)
    args=ap.parse_args();args.base_out=args.root/'pilot';args.manifest=args.root/'pilot-selection-v1.json'
    if args.out.exists():raise FileExistsError(args.out)
    base.policy();started=time.monotonic()
    metadata,selection,initial,preflight,manifest,pools=screen.authenticate_inputs(args)
    paths={1000:args.root/'current-lowrate-continue1000-v1/checkpoint-step1000.pt',
           **{st:args.root/f'current-lowrate-fresh5000-v1/checkpoint-step{st}.pt' for st in (4500,5000)}}
    payloads={};hashes={}
    for step,path in paths.items():
        receipt=json.loads(path.with_suffix('.json').read_text());hashes[step]=base.sha(path)
        if hashes[step]!=receipt['checkpoint_sha256'] or receipt['step']!=step:raise ValueError('Checkpoint receipt differs')
        payloads[step]=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if payloads[step]['step']!=step or payloads[step]['identity']['channel_selection']!=selection:raise ValueError('Checkpoint selection differs')
    teacher=base.FrozenAudioVAE2.from_files(args.assets/'audio_vae_v2.py',args.assets/'audiovae.pth',device='cuda')
    model=base.build_student(teacher.model.decoder,selection['stage2_indices'],selection['stage3_indices'])
    frozen=screen.frozen_versions(model);teacher_before={k:(v.data_ptr(),v._version) for k,v in teacher.state_dict().items()}
    report={'version':VERSION,'script_sha256':base.sha(__file__),'checkpoint_sha256':hashes,'sources':[],
        'teacher_parameters':parameter_audit(teacher.model.decoder),'student_parameters':{},
        'runtime':{'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'dtype':'float32','batch_size':1},
        'parameter_updates':0,'optimizer_created':False,
        'interpretation':'Hidden DC is each channel mean error across the stated region, not a waveform DC offset. Near-silence masks are exact teacher-defined20ms windows. Internal selected coordinates need not preserve learned representations.'}
    by_id={c['source_id']:c for c in pools['development']}
    with torch.no_grad():
        for sid in SOURCES:
            crop=by_id[sid];z,target,valid,spans=base.batch([crop]);regions=masks(target,valid)
            base.teacher_forward(teacher,z)
            tr,tv=capture(teacher.model.decoder,z);repeated=base.teacher_forward(teacher,z)
            if not all(torch.equal(tr[k],repeated[k]) for k in ('group_input','group_output','waveform')):raise RuntimeError('Teacher trace unstable')
            source={'source_id':sid,'geometry':{k:crop[k] for k in ('start_frame','context_start_frame','context_frames','valid_scored_samples')},
                'target_consistency':target_comparison(tr,target,valid,regions),'checkpoints':{}}
            for step,payload in payloads.items():
                model.load_group_state_dict(payload['group']);initial_digest=screen.group_digest(model)
                for _ in range(2):model.forward_latents(z)
                sr,sv=capture(model.decoder,z)
                if not torch.equal(sr['group_input'],tr['group_input']):raise RuntimeError('Frozen prefix differs')
                row={'layers':layer_report(sv,tv,regions,selection),
                     'waveform':{name:stats(sr['waveform'],target,mask) for name,mask in regions.items()},
                     'quiet_windows':wave_windows(sr['waveform'],target,valid)}
                # Exact shared-boundary substitution tests the frozen suffix on original teacher inputs.
                teacher_suffix=model.suffix_from_group(tr['group_output'])
                row['teacher_group_through_student_suffix']=target_comparison({'waveform':teacher_suffix},tr['waveform'],valid,regions)
                if sid in SOURCES[:2]:
                    halfway=model.suffix_from_group(tr['group_output']+.5*(sr['group_output']-tr['group_output']))
                    row['half_group_error_intervention']={'waveform':{name:stats(halfway,target,mask) for name,mask in regions.items()},
                         'quiet_windows':wave_windows(halfway,target,valid)}
                source['checkpoints'][str(step)]=row
                if str(step) not in report['student_parameters']:report['student_parameters'][str(step)]=parameter_audit(model.decoder)
                if screen.group_digest(model)!=initial_digest:raise RuntimeError('Read-only diagnostic changed weights')
                del sr,sv,teacher_suffix
            report['sources'].append(source);del tr,tv,repeated,z,target,valid
            base.event('layer_diagnosis_source',source_id=sid,completed=len(report['sources']))
        if args.fresh_panel:
            panel=json.loads(args.fresh_panel.read_text());report['fresh_targets']=[]
            for selected in panel['selected']:
                path=Path(selected['shard'])/'pairs.pt';receipt=json.loads(path.with_name('receipt.json').read_text())
                if base.sha(path)!=receipt['pairs_sha256']:raise RuntimeError('Fresh cache hash changed')
                saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True);crop=saved['crops'][selected['index']%300]
                if crop['source_id']!=selected['source_id']:raise ValueError('Fresh source selection differs')
                z,target,valid,_=base.batch([crop]);regions=masks(target,valid);tr=base.teacher_forward(teacher,z)
                report['fresh_targets'].append({'source_id':crop['source_id'],
                    'geometry':{k:crop[k] for k in ('start_frame','context_start_frame','context_frames','valid_scored_samples')},
                    'comparison':target_comparison(tr,target,valid,regions)})
                del saved,tr,z,target,valid
    report['preserved']={'teacher':teacher_before=={k:(v.data_ptr(),v._version) for k,v in teacher.state_dict().items()},
        'frozen_student':frozen==screen.frozen_versions(model),'checkpoint_files':all(base.sha(p)==hashes[s] for s,p in paths.items())}
    if not all(report['preserved'].values()):raise RuntimeError('Preservation check failed')
    report['elapsed_seconds']=time.monotonic()-started;base.write_json(args.out,report)
    base.event('layer_diagnosis_completed',out=str(args.out),seconds=report['elapsed_seconds'],preserved=report['preserved'])

if __name__=='__main__':main()
