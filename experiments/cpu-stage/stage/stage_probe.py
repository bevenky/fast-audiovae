"""Explicit CPU-only extraction and paired standalone-stage timing hooks.

Nothing runs on import. Run extract once with the real L170 latent, then bench
serially on the target CPU. Compare the same BCT arrays, weights, thread count
and preallocated ORT input/output boundary; this is not a full-decoder RTF.
"""
from __future__ import annotations
import argparse,copy,gc,hashlib,json,os,platform,random,statistics,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',HIP_VISIBLE_DEVICES='-1',
                  ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from check_stage import compare

GEOMETRY={256:40800,128:81600,64:163200,32:326400}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def options(native,stage=None):
    import onnxruntime as ort
    if ort.__version__!='1.29.0':raise RuntimeError('ORT1.29.0 required')
    so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1
    so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;so.log_severity_level=3
    so.add_session_config_entry('session.intra_op.allow_spinning','0')
    so.add_session_config_entry('session.inter_op.allow_spinning','0')
    so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.register_custom_ops_library(str(native.resolve()))
    if stage:so.register_custom_ops_library(str(stage.resolve()))
    return so
def session(m,native,stage=None):
    import onnxruntime as ort
    s=ort.InferenceSession(m.SerializeToString(),options(native,stage),providers=['CPUExecutionProvider'])
    if s.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('Unexpected provider')
    return s
def standalone(source,candidate,record,backend,q,parts,mode,isa,debug=False,reference_fusion='none'):
    if reference_fusion not in ('none','both'):raise ValueError('reference_fusion must be none or both')
    c=record['channels'];t=GEOMETRY[c];sinput=record['input'];outputs=record['unit_outputs'] if debug else [record['output']]
    info=lambda v:helper.make_tensor_value_info(v,TensorProto.FLOAT,[1,c,t])
    native_nodes=[copy.deepcopy(n) for n in source.graph.node if n.name in record['removed_nodes']]
    # All selected Reshapes have been proved identity BCT views. Fixed geometry
    # avoids importing unrelated decoder shape programs into the stage graph.
    shape_name='sp_probe_fixed_shape'
    for node in native_nodes:
        if node.op_type=='Reshape':node.input[1]=shape_name
        if node.domain=='venky.audio.cpu.portable':
            for attr in node.attribute:
                if attr.name=='backend':attr.i=backend
    needed={v for n in native_nodes for v in n.input}
    ini=[copy.deepcopy(v) for v in source.graph.initializer if v.name in needed]
    ini.append(numpy_helper.from_array(np.array([1,c,t],np.int64),shape_name))
    ref=helper.make_model(helper.make_graph(native_nodes,'native_three_unit_stage',[info(sinput)],[info(v) for v in outputs],ini),
              opset_imports=list(source.opset_import),ir_version=source.ir_version)
    if reference_fusion=='both':
        # Fuse the complete comparison control before creating either debug or
        # timed sessions. Keep the same forced arithmetic/sine backend as the
        # stage candidate, and fail unless all three proven units were fused.
        from fast_audiovae.graph.block_fusion import rewrite_model
        ref,_=rewrite_model(ref,mode='both',backend=backend,expected_chains=3,expected_adds=3)
    node=next(copy.deepcopy(n) for n in candidate.graph.node if n.name==record['node'])
    values={'tile_time':q,'segments':parts,'backend':backend,'matrix_mode':mode,'matrix_isa':isa}
    for attr in node.attribute:
        if attr.name in values:attr.i=values[attr.name]
    if debug:node.op_type='StageStackDebugF32';del node.output[:];node.output.extend(outputs)
    needed=set(node.input);ini=[copy.deepcopy(v) for v in candidate.graph.initializer if v.name in needed]
    cand=helper.make_model(helper.make_graph([node],'candidate_stage',[info(sinput)],[info(v) for v in outputs],ini),
              opset_imports=list(candidate.opset_import),ir_version=candidate.ir_version)
    onnx.checker.check_model(ref);onnx.checker.check_model(cand);return ref,cand
def bound(s,x):
    import onnxruntime as ort
    y=np.empty_like(x);binding=s.io_binding();xv=ort.OrtValue.ortvalue_from_numpy(x);yv=ort.OrtValue.ortvalue_from_numpy(y)
    binding.bind_ortvalue_input(s.get_inputs()[0].name,xv);binding.bind_ortvalue_output(s.get_outputs()[0].name,yv)
    return lambda:s.run_with_iobinding(binding),(s,binding,xv,yv,x,y),y

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('extract','bench'))
    p.add_argument('--source',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--native-library',required=True,type=Path);p.add_argument('--stage-library',type=Path)
    p.add_argument('--folder',required=True,type=Path);p.add_argument('--cases',type=Path)
    p.add_argument('--uid',required=True);p.add_argument('--channels',default='256')
    p.add_argument('--latent-frames',type=int,help='Explicit prefix length for capture; must be 170 for this geometry')
    p.add_argument('--tiles',default='64,128,256');p.add_argument('--backends',default='4,5')
    p.add_argument('--matrix-mode',type=int,default=0);p.add_argument('--segments',type=int,default=2)
    p.add_argument('--reference-fusion',choices=('none','both'),default='none',
                   help='Compare against original native operators or all3 proved chain+add fusions on the same backend')
    p.add_argument('--warmups',type=int,default=2);p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--output',type=Path);p.add_argument('--affinity',help='Optional explicit CPU IDs, e.g. 0,1')
    a=p.parse_args()
    if a.affinity:os.sched_setaffinity(0,{int(v) for v in a.affinity.split(',')})
    channels=[int(v) for v in a.channels.split(',')];a.folder.mkdir(parents=True,exist_ok=True)
    audit=json.loads(a.candidate.with_suffix('.stage.json').read_text())
    if sha(a.source)!=audit['source_sha256'] or sha(a.candidate)!=audit['output_sha256']:raise RuntimeError('Source/candidate hash mismatch')
    records=[r for r in audit['changes'] if r['channels'] in channels]
    if len(records)!=len(channels):raise ValueError('Candidate does not contain every selected stage')
    source=onnx.load(a.source);candidate=onnx.load(a.candidate)
    if a.command=='extract':
        if not a.cases:raise ValueError('--cases required')
        z=np.load(a.cases,allow_pickle=False)[a.uid+'__z']
        original_frames=z.shape[-1]
        if a.latent_frames is not None:
            if a.latent_frames!=170 or original_frames<a.latent_frames:
                raise ValueError('Capture prefix must be 170 frames from a sufficiently long real case')
            z=np.ascontiguousarray(z[...,:a.latent_frames])
        if z.dtype!=np.float32 or z.shape!=(1,64,170):raise ValueError('Exact real FP32 L170 latent required for stated geometry')
        for r in records:
            c=r['channels'];m=copy.deepcopy(source);del m.graph.output[:]
            names=[r['input'],*r['unit_outputs']]
            m.graph.output.extend(helper.make_tensor_value_info(v,TensorProto.FLOAT,[1,c,GEOMETRY[c]]) for v in names)
            s=session(m,a.native_library);values=s.run(None,{s.get_inputs()[0].name:z});hashes={}
            for i,v in enumerate(values):
                if v.shape!=(1,c,GEOMETRY[c]) or v.dtype!=np.float32:raise RuntimeError('Unexpected captured stage geometry')
                file=a.folder/f'c{c}_{"input" if i==0 else "unit"+str(i)}.npy'
                np.save(file,v);hashes[file.name]=sha(file)
            manifest={'c':c,'t':GEOMETRY[c],'source_sha256':sha(a.source),'native_library_sha256':sha(a.native_library),
                      'cases_sha256':sha(a.cases),'uid':a.uid,'original_latent_frames':original_frames,
                      'explicit_prefix_frames':a.latent_frames,'latent_sha256':hashlib.sha256(z.tobytes()).hexdigest(),
                      'files':hashes,'gpu_used':False,'timed':False,'ort':'1.29.0'}
            (a.folder/f'c{c}_capture.json').write_text(json.dumps(manifest,indent=2)+'\n');del s,m,values;gc.collect()
        return
    if not a.stage_library or not a.output:raise ValueError('--stage-library and --output required for bench')
    report={'scope':'Isolated complete native three-unit residual stage, not full decoder RTF','rows':[],
            'native_library_sha256':sha(a.native_library),'stage_library_sha256':sha(a.stage_library),
            'source_sha256':sha(a.source),'candidate_sha256':sha(a.candidate),'gpu_used':False,'ort':'1.29.0',
            'reference_fusion':a.reference_fusion,
            'threads':2,'inter_threads':1,'platform':platform.platform(),
            'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'protocol':{'serial_randomized_AB':True,'preallocated_IO':True,'warmups':a.warmups,'repeats':a.repeats,'spinning':False}}
    rng=random.Random(14027)
    def save():a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    for r in records:
        c=r['channels'];manifest=json.loads((a.folder/f'c{c}_capture.json').read_text())
        if manifest['source_sha256']!=sha(a.source):raise RuntimeError('Capture source changed')
        for f,h in manifest['files'].items():
            if sha(a.folder/f)!=h:raise RuntimeError('Capture array changed')
        x=np.load(a.folder/f'c{c}_input.npy',allow_pickle=False)
        expected=[np.load(a.folder/f'c{c}_unit{u+1}.npy',allow_pickle=False) for u in range(3)]
        for backend in [int(v) for v in a.backends.split(',')]:
            isa=0 if a.matrix_mode else (256 if backend==4 else 512)
            for q in [int(v) for v in a.tiles.split(',')]:
                ref,probe=standalone(source,candidate,r,backend,q,a.segments,a.matrix_mode,isa,True,a.reference_fusion)
                rs=session(ref,a.native_library);cs=session(probe,a.native_library,a.stage_library)
                ry=rs.run(None,{r['input']:x});cy=cs.run(None,{r['input']:x})
                validation=[{'original_capture':compare(ry[u],expected[u],f'C{c} captured unit{u}'),
                             'candidate_native':compare(cy[u],ry[u],f'C{c} candidate unit{u}'),
                             'candidate_original_capture':compare(cy[u],expected[u],f'C{c} candidate captured unit{u}')} for u in range(3)]
                del rs,cs,ry,cy;gc.collect()
                ref,probe=standalone(source,candidate,r,backend,q,a.segments,a.matrix_mode,isa,False,a.reference_fusion)
                methods={};holds=[];out={}
                for name,m in [('native',ref),('candidate',probe)]:
                    fn,hold,y=bound(session(m,a.native_library,a.stage_library),x);methods[name]=fn;holds.append(hold);out[name]=y
                for fn in methods.values():fn()
                compare(out['candidate'],out['native'],'timed final')
                for _ in range(a.warmups):
                    order=list(methods);rng.shuffle(order)
                    for name in order:methods[name]()
                samples={name:[] for name in methods};orders=[];enabled=gc.isenabled();gc.disable()
                try:
                    for _ in range(a.repeats):
                        order=list(methods);rng.shuffle(order);orders.append(order)
                        for name in order:
                            start=time.perf_counter_ns();methods[name]();samples[name].append(time.perf_counter_ns()-start)
                finally:
                    if enabled:gc.enable()
                medians={name:statistics.median(v) for name,v in samples.items()}
                report['rows'].append({'c':c,'t':GEOMETRY[c],'q':q,'backend':backend,'matrix_mode':a.matrix_mode,'matrix_isa':isa,
                         'reference_fusion':a.reference_fusion,
                         'reference_graph_sha256':hashlib.sha256(ref.SerializeToString()).hexdigest(),
                         'candidate_graph_sha256':hashlib.sha256(probe.SerializeToString()).hexdigest(),
                         'segments':a.segments,'validation':validation,'samples_ns':samples,'orders':orders,'median_ns':medians,
                         'stage_speedup':medians['native']/medians['candidate'],
                         'stage_seconds_per_audio_second':{name:ns/1e9/6.8 for name,ns in medians.items()}})
                save();print(json.dumps(report['rows'][-1]),flush=True);del methods,holds,out,fn,hold;gc.collect()
    report['status']='complete';save()

if __name__=='__main__':main()
