"""Locate frozen AudioVAE2 encoder batch divergence without changing its weights.

Only row4 of the historical batch is retained, capped to2048 time positions per
leaf input/output. All scalar comparisons and local references are diagnostic;
this does not change cache contents, teacher semantics or training state.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

PREFIX=2048

def comparison(actual,reference,atol=1e-5,rtol=1e-4):
    a=np.asarray(actual);b=np.asarray(reference)
    if a.shape!=b.shape or a.ndim!=3:raise ValueError('Comparison shape mismatch or non-BCT capture')
    if a.dtype!=np.float32 or b.dtype!=np.float32:raise TypeError('Expected FP32 captured values')
    if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Empty or nonfinite capture')
    e=a.astype(np.float64)-b.astype(np.float64);abs_e=np.abs(e)
    failed=abs_e>atol+rtol*np.abs(b.astype(np.float64))
    return {'shape':list(a.shape),'samples':int(a.size),'bitwise_equal':bool(np.array_equal(a.view(np.uint32),b.view(np.uint32))),
        'within_tolerance':not bool(failed.any()),'atol':atol,'rtol':rtol,'failed_samples':int(failed.sum()),
        'max_abs_error':float(abs_e.max()),'rms_error':float(np.sqrt(np.mean(e*e))),
        'actual_rms':float(np.sqrt(np.mean(a.astype(np.float64)**2))),
        'reference_rms':float(np.sqrt(np.mean(b.astype(np.float64)**2))),
        'max_error_index':[int(v) for v in np.unravel_index(int(abs_e.argmax()),a.shape)],
        'per_time_max_abs':abs_e.reshape(-1,a.shape[-1]).max(axis=0).tolist(),
        'per_channel_max_abs':abs_e.max(axis=(0,2)).tolist()}

def prefix_output_limit(input_samples,kernel,dilation,stride,left_padding,requested):
    if min(input_samples,kernel,dilation,stride,requested)<1 or left_padding<0:raise ValueError('Invalid convolution geometry')
    # No artificial right padding may enter a replayed output.
    available=(input_samples+left_padding-dilation*(kernel-1)-1)//stride+1
    return max(0,min(requested,available))

def capture_array(tensor):return tensor.detach().cpu().contiguous().numpy()

def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def leaf_modules(model):
    return [(name,m) for name,m in model.named_modules() if name.startswith('encoder.') and not any(m.children())]

def weight_info(module,tensor_hash):
    import torch
    w=getattr(module,'weight',None)
    info={'hooks':[{'type':type(h).__name__,'name':getattr(h,'name',None),'dim':getattr(h,'dim',None)} for h in module._forward_pre_hooks.values()]}
    if isinstance(w,torch.Tensor):
        info.update(effective_weight_sha256=tensor_hash(w),effective_weight_shape=list(w.shape),effective_weight_dtype=str(w.dtype),effective_weight_device=str(w.device),
            effective_weight_registered_parameter='weight' in module._parameters,
            effective_weight_registered_buffer='weight' in module._buffers)
    for attr in ('weight_g','weight_v','bias','alpha'):
        t=getattr(module,attr,None)
        if isinstance(t,torch.Tensor):info[attr+'_sha256']=tensor_hash(t)
    return info

def trace_once(teacher,x,row,tensor_hash):
    import torch
    captures={};order=[];handles=[]
    def hook(name):
        def observe(module,args,output):
            if name in captures:raise RuntimeError('Encoder leaf was reused unexpectedly: '+name)
            if not args or not isinstance(args[0],torch.Tensor) or not isinstance(output,torch.Tensor):raise TypeError('Unexpected encoder leaf signature: '+name)
            a=args[0];b=output
            if a.ndim!=3 or b.ndim!=3 or a.shape[0]!=x.shape[0] or b.shape[0]!=x.shape[0]:raise ValueError('Unexpected leaf tensor geometry')
            if a.dtype!=torch.float32 or b.dtype!=torch.float32:raise TypeError('Unexpected non-FP32 leaf')
            captures[name]={'input':a[row:row+1,...,:PREFIX].detach().cpu().contiguous().clone(),
                'output':b[row:row+1,...,:PREFIX].detach().cpu().contiguous().clone(),
                'input_shape':list(a.shape),'output_shape':list(b.shape),'type':type(module).__name__,
                # A forward hook runs after the original weight-normalization
                # pre-hook and the convolution that consumed its result.
                'weight':weight_info(module,tensor_hash),'on_mu_path':not name.startswith('encoder.fc_logvar')}
            order.append(name)
        return observe
    try:
        for name,module in leaf_modules(teacher.model):handles.append(module.register_forward_hook(hook(name)))
        before=tensor_hash(x);z=teacher.encode(x).detach().cpu()
        if tensor_hash(x)!=before:raise RuntimeError('Encoder modified input')
    finally:
        for handle in handles:handle.remove()
    return z,captures,order

def reference_leaf(module,values,expected_weight,tensor_hash,requested=64):
    """Optional bounded reference on captured row inputs, not a model rewrite.

    FP32 eager replay remains on the original device. FP64 is a CPU arithmetic
    reference with identical coefficients promoted to double. Different input
    length/backend may alter rounding; these are diagnostics, not pass gates.
    """
    import torch
    import torch.nn.functional as F
    if weight_info(module,tensor_hash)!=expected_weight:raise RuntimeError('Effective leaf parameters/hooks differ before replay')
    actual=values['output'];source=values['input']
    result={'diagnostic_only':True,'original_full_batch_shape':values['input_shape'],
        'replay_batch_size':1,'reference_outputs_max':requested,
        'limitation':'Selected-row bounded-prefix replay changes execution geometry; FP64 reference is not a bitwise parity contract'}
    if isinstance(module,torch.nn.Conv1d):
        if not hasattr(module,'_CausalConv1d__padding'):raise TypeError('Unknown encoder convolution padding contract')
        left=module._CausalConv1d__padding*2-module._CausalConv1d__output_padding
        k=module.kernel_size[0];d=module.dilation[0];s=module.stride[0]
        n=prefix_output_limit(source.shape[-1],k,d,s,left,min(requested,actual.shape[-1]))
        if n<1:raise ValueError('Insufficient authentic prefix for convolution reference')
        needed=max(1,(n-1)*s-left+d*(k-1)+1)
        src=source[...,:needed].contiguous();w=module.weight.detach();bias=module.bias.detach() if module.bias is not None else None
        y32=F.conv1d(F.pad(src.to(w.device),(left,0)),w,bias,stride=s,dilation=d,groups=module.groups)[...,:n]
        y64=F.conv1d(F.pad(src.double(),(left,0)),w.cpu().double(),bias.cpu().double() if bias is not None else None,stride=s,dilation=d,groups=module.groups)[...,:n]
        result['geometry']={'left_padding':left,'kernel':k,'dilation':d,'stride':s,'groups':module.groups,'source_prefix_samples':needed}
    elif type(module).__name__=='Snake1d':
        n=min(requested,source.shape[-1],actual.shape[-1]);src=source[...,:n];alpha=module.alpha.detach()
        def eager(x,a):return x+(a+1e-9).reciprocal()*torch.sin(a*x).pow(2)
        y32=eager(src.to(alpha.device),alpha);y64=eager(src.double(),alpha.cpu().double())
        result['geometry']={'pointwise':True,'source_prefix_samples':n}
    else:raise TypeError('Unsupported reference leaf '+type(module).__name__)
    expected=actual[...,:n]
    result['original_vs_eager_fp32']=comparison(capture_array(expected),capture_array(y32))
    result['original_vs_fp64_rounded_fp32']=comparison(capture_array(expected),capture_array(y64.float()))
    result['eager_fp32_vs_fp64_rounded_fp32']=comparison(capture_array(y32),capture_array(y64.float()))
    result['scored_outputs']=n
    return result

def main(args):
    import torch
    import torch.nn.functional as F
    import cache_probe
    import diagnostic_common
    import repair_natural_history
    from diagnostic_common import load_context,atomic_json,status
    from repair_natural_history import source_rows,authentic_audio,tensor_hash
    from audiovae_student.objective_comparison import state_fingerprint
    from audiovae_student.fusion_evaluation import _preserved_evaluation
    if args.out.exists():raise RuntimeError('Refusing to overwrite trace directory')
    if args.row!=4:raise ValueError('This bounded trace is pinned to original batch row4')
    if args.warmups!=2:raise ValueError('This trace requires two unhooked warmups per shape')
    ctx=load_context();rows,pins=source_rows(ctx)
    audios,provenance=zip(*(authentic_audio(rows[sid]) for sid in cache_probe.IDS))
    if [a.shape[-1] for a in audios]!=cache_probe.LENGTHS:raise ValueError('Historical lengths changed')
    cpu_batch=torch.cat([F.pad(a,(0,93440-a.shape[-1])) for a in audios]);cpu_single=cpu_batch[args.row:args.row+1].contiguous().clone()
    if not torch.equal(cpu_batch[args.row:args.row+1].view(torch.int32),cpu_single.view(torch.int32)):raise ValueError('Input row differs')
    teacher=ctx.teacher();before=state_fingerprint(teacher.model.state_dict())
    report={'format_version':1,'scope':__doc__,'row':args.row,'source_id':cache_probe.IDS[args.row],
        'historical_ids':cache_probe.IDS,'historical_lengths':cache_probe.LENGTHS,'source_provenance':list(provenance),'source_manifest_pins':pins,
        'batch_input_sha256':tensor_hash(cpu_batch),'single_input_sha256':tensor_hash(cpu_single),'prefix_cap':PREFIX,
        'warmups_per_shape':args.warmups,'startup_order':['construct_fresh_teacher_without_encode','batch8_warmup1','batch8_warmup2','batch8_trace','single_padded_warmup1','single_padded_warmup2','single_padded_trace'],
        'startup_caveat':'Fresh process required. B8 is deliberately the first encode shape; no generic teacher warmup is performed. This preserves the observed process-history-dependent failure. Optional references run last.',
        'teacher_state_before':before,'teacher':teacher.provenance,
        'sources':{str(Path(m.__file__)):sha_file(m.__file__) for m in (cache_probe,diagnostic_common,repair_natural_history)},
        'trace_source_sha256':sha_file(__file__),'parameter_updates':0,'optimizer_updates':0,
        'runtime':{'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_tf32':torch.backends.cudnn.allow_tf32,'deterministic':torch.are_deterministic_algorithms_enabled()}}
    modes={};args.out.mkdir(parents=True)
    with _preserved_evaluation(teacher.model):
        for label,cpu_x,row in [('batch8',cpu_batch,args.row),('single_padded',cpu_single,0)]:
            status('layer_trace',mode=label)
            x=cpu_x.to(teacher.device)
            for _ in range(args.warmups):warm=teacher.encode(x).detach().cpu()
            z,captures,order=trace_once(teacher,x,row,tensor_hash)
            modes[label]={'z':z[row:row+1].contiguous(),'captures':captures,'order':order,
                'hooked_vs_unhooked':comparison(capture_array(z[row:row+1]),capture_array(warm[row:row+1]))}
            del x,warm,z
        batch=modes['batch8'];single=modes['single_padded']
        if batch['order']!=single['order']:raise RuntimeError('Different encoder execution topology')
        report['hooked_vs_unhooked']={k:v['hooked_vs_unhooked'] for k,v in modes.items()}
        report['hooked_execution_parity_passed']=all(v['within_tolerance'] for v in report['hooked_vs_unhooked'].values())
        if not report['hooked_execution_parity_passed']:
            atomic_json(args.out/'hook-parity-failure.json',report)
            raise RuntimeError('Hooks changed output beyond the existing tolerance; trace is not interpretable')
        report['final_latents_batch_vs_single']=comparison(capture_array(batch['z']),capture_array(single['z']))
        layers=[]
        for name in batch['order']:
            a=batch['captures'][name];b=single['captures'][name]
            layers.append({'name':name,'type':a['type'],'on_mu_path':a['on_mu_path'],
                'batch_input_shape':a['input_shape'],'single_input_shape':b['input_shape'],
                'batch_output_shape':a['output_shape'],'single_output_shape':b['output_shape'],
                'input':comparison(capture_array(a['input']),capture_array(b['input'])),
                'output':comparison(capture_array(a['output']),capture_array(b['output'])),
                'effective_weight_and_hooks_equal':a['weight']==b['weight'],'batch_weight':a['weight'],'single_weight':b['weight']})
        report['layers']=layers
        report['first_bitwise_difference']=next((v['name'] for v in layers if v['on_mu_path'] and not v['output']['bitwise_equal']),None)
        report['first_tolerance_failure']=next((v['name'] for v in layers if v['on_mu_path'] and not v['output']['within_tolerance']),None)
        report['first_output_max_abs_over_1e_3']=next((v['name'] for v in layers if v['on_mu_path'] and v['output']['max_abs_error']>1e-3),None)
        if args.replay_first and report['first_tolerance_failure']:
            name=report['first_tolerance_failure'];module=dict(teacher.model.named_modules())[name]
            report['first_divergence_references']={'node':name,'modes':{}}
            for label,m in modes.items():
                cap=m['captures'][name]
                report['first_divergence_references']['modes'][label]=reference_leaf(module,cap,cap['weight'],tensor_hash)
    report['teacher_state_after']=state_fingerprint(teacher.model.state_dict())
    if report['teacher_state_after']!=before:raise RuntimeError('Frozen registered model state changed')
    report['checkpoint_preservation']=ctx.verify_files()
    captured=args.out/'captured-prefixes.pt';torch.save(modes,captured)
    report['capture_file']={'name':captured.name,'sha256':sha_file(captured),'bytes':captured.stat().st_size,
        'scope':'Only selected row input/output prefixes, with final selected-row latents; no source PCM or full weight tensors'}
    atomic_json(args.out/'trace.json',report)
    status('layer_trace_complete',first=report['first_tolerance_failure'],result=str(args.out/'trace.json'))
    return report

if __name__=='__main__':
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--row',type=int,default=4);p.add_argument('--warmups',type=int,default=2);p.add_argument('--replay-first',action='store_true')
    a=p.parse_args();lock=Path('/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock')
    with lock.open('rb') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);main(a)
