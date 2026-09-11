"""Replay only the isolated fc_mu convolution with exact captured input/weights.

Run in a fresh process. No complete encoder forward, cache regeneration or
checkpoint update is performed. The large trace archive is read with mmap;
only four small fc_mu row tensors are copied and retained.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from trace_encoder import comparison,weight_info,capture_array

NODE='encoder.fc_mu'

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def channel_summary(actual,reference):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1]!=64:raise ValueError('Expected BCT64 output')
    e=a-b
    return {'per_channel_mean_error':e.mean(axis=(0,2)).tolist(),
        'per_channel_rms_error':np.sqrt(np.mean(e*e,axis=(0,2))).tolist(),
        'per_channel_max_abs_error':np.abs(e).max(axis=(0,2)).tolist(),
        'channel63_actual':a[0,63].tolist(),'channel63_reference':b[0,63].tolist(),
        'channel63_error':e[0,63].tolist()}

def main(args):
    import torch
    import torch.nn.functional as F
    from diagnostic_common import load_context,atomic_json,status
    from repair_natural_history import tensor_hash
    from audiovae_student.objective_comparison import state_fingerprint
    from audiovae_student.fusion_evaluation import _preserved_evaluation
    if args.out.exists():raise RuntimeError('Refusing to overwrite focused replay results')
    trace=json.loads(args.trace_json.read_text());node=next(v for v in trace['layers'] if v['name']==NODE)
    if trace['first_tolerance_failure']!=NODE or not node['input']['bitwise_equal'] or not node['effective_weight_and_hooks_equal']:
        raise ValueError('Trace no longer identifies identical-input/effective-weight fc_mu divergence')
    if args.captures.stat().st_size!=trace['capture_file']['bytes'] or file_sha(args.captures)!=trace['capture_file']['sha256']:
        raise ValueError('Captured prefix archive differs from the completed trace')
    # mmap avoids materializing the other encoder layers or copying the archive.
    saved=torch.load(args.captures,map_location='cpu',weights_only=True,mmap=True)
    selected={}
    for mode in ('batch8','single_padded'):
        v=saved[mode]['captures'][NODE]
        selected[mode]={'input':v['input'].contiguous().clone(),'output':v['output'].contiguous().clone(),'weight':v['weight']}
    del saved
    xcpu=selected['batch8']['input'];xcpu_single=selected['single_padded']['input']
    if xcpu.shape!=(1,2048,146) or not torch.equal(xcpu.view(torch.int32),xcpu_single.view(torch.int32)):
        raise ValueError('Expected the complete matching146-frame captured fc_mu input')
    if any(v['output'].shape!=(1,64,146) for v in selected.values()):raise ValueError('Wrong captured head output shape')
    ctx=load_context();initial_enabled=args.initial_cudnn=='on';torch.backends.cudnn.enabled=initial_enabled
    teacher=ctx.teacher();head=teacher.model.encoder.fc_mu
    if head.kernel_size!=(3,) or head.stride!=(1,) or head.dilation!=(1,) or head.groups!=1:
        raise ValueError('Unexpected fc_mu geometry')
    left=head._CausalConv1d__padding*2-head._CausalConv1d__output_padding
    if left!=2:raise ValueError('Unexpected causal left padding')
    before=state_fingerprint(head.state_dict());results={}
    x1=xcpu.to(teacher.device);x8=x1.repeat(8,1,1).contiguous()
    inputs={'single':x1,'batch8_repeated_row':x8}
    first=['batch8_repeated_row','single'] if args.first_shape=='batch8' else ['single','batch8_repeated_row']
    report={'format_version':1,'scope':__doc__,'trace_json_sha256':file_sha(args.trace_json),
        'captures_sha256':trace['capture_file']['sha256'],'source_sha256':file_sha(__file__),
        'input_sha256':tensor_hash(xcpu),'head_state_before':before,'initial_cudnn':initial_enabled,'first_shapes':first,
        'geometry':{'input':[1,2048,146],'batch_input':[8,2048,146],'weight':[64,2048,3],'output':[1,64,146],'left_padding':left},
        'batch_data_policy':'Eight repeated copies of the exact captured row, not the seven uncaptured peers; mathematical convolution is independent across batch rows',
        'teacher':teacher.provenance,'full_encoder_calls':0,'parameter_updates':0,'optimizer_updates':0,
        'runtime':{'torch':str(torch.__version__),'cudnn':torch.backends.cudnn.version(),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_tf32':torch.backends.cudnn.allow_tf32,'deterministic':torch.are_deterministic_algorithms_enabled()}}
    args.out.mkdir(parents=True)
    with _preserved_evaluation(head):
        # Normal module invocation refreshes WN in its original pre-hook. These
        # are the first convolution calls made by this fresh process.
        for shape in first:
            y=head(inputs[shape]).detach().cpu()
            results['normal_initial_'+shape]=y[:1].contiguous().clone()
            report.setdefault('normal_module_row_consistency',{})[shape]=comparison(capture_array(y),capture_array(y[:1].expand_as(y).contiguous()))
        effective=weight_info(head,tensor_hash)
        if effective!=node['single_weight'] or effective!=node['batch_weight']:
            raise RuntimeError('Actual post-hook effective head weights differ from captured weights')
        report['effective_weights']=effective
        # Frozen detached FP32 operands, without invoking any module or hook.
        w=head.weight.detach().contiguous().clone();b=head.bias.detach().contiguous().clone()
        report['bias63']=float(b[63])
        for enabled in (True,False):
            with torch.backends.cudnn.flags(enabled=enabled,benchmark=False,deterministic=True,allow_tf32=False):
                for shape in first:
                    padded=F.pad(inputs[shape],(left,0))
                    for repeat in range(2):
                        key=f'functional_cudnn_{"on" if enabled else "off"}_{shape}_repeat{repeat}'
                        y=F.conv1d(padded,w,b,stride=1,padding=0,dilation=1,groups=1)
                        results[key]=y[:1].detach().cpu().contiguous().clone()
        # Full146-frame FP64 CPU reference from exactly the same FP32 values.
        torch.set_num_threads(1)
        with torch.backends.mkldnn.flags(enabled=False):
            reference=F.conv1d(F.pad(xcpu.double(),(left,0)),w.cpu().double(),b.cpu().double(),stride=1,padding=0,dilation=1,groups=1)
        if reference.shape!=(1,64,146) or not bool(torch.isfinite(reference).all()):raise RuntimeError('Invalid FP64 reference')
        ref32=reference.float();report['fp64_reference_sha256']=hashlib.sha256(reference.contiguous().numpy().astype('<f8',copy=False).tobytes()).hexdigest()
        report['reference_policy']='CPU FP64 functional convolution with original FP32 input, effective weights and bias promoted to double; comparisons use one final FP32 rounding'
        all_values={**{f'captured_{mode}':v['output'] for mode,v in selected.items()},**results}
        report['comparisons']={}
        for name,value in all_values.items():
            a=capture_array(value);r=capture_array(ref32)
            report['comparisons'][name]={'vs_fp64_rounded':comparison(a,r),'channel_details':channel_summary(a,capture_array(reference)),
                'vs_captured_batch':comparison(a,capture_array(selected['batch8']['output'])),
                'vs_captured_single':comparison(a,capture_array(selected['single_padded']['output']))}
        if weight_info(head,tensor_hash)!=effective:raise RuntimeError('Effective weights changed during focused replay')
    report['head_state_after']=state_fingerprint(head.state_dict())
    if report['head_state_after']!=before:raise RuntimeError('Head parameters changed')
    report['checkpoint_preservation']=ctx.verify_files()
    report['focused_module_calls']=2;report['focused_functional_fp32_calls']=8;report['focused_cpu_fp64_calls']=1
    atomic_json(args.out/'fc-mu-reference.json',report)
    status('fc_mu_reference_complete',output=str(args.out/'fc-mu-reference.json'))
    return report

if __name__=='__main__':
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace-json',type=Path,required=True);p.add_argument('--captures',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--initial-cudnn',choices=('on','off'),default='on');p.add_argument('--first-shape',choices=('batch8','single'),default='batch8')
    a=p.parse_args();lock=Path('/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock')
    with lock.open('rb') as h:
        fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);main(a)
