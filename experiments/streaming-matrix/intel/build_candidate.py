"""Build and derive the isolated streaming projection candidate, no inference."""
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ort-include',type=Path,required=True);p.add_argument('--core',type=Path,required=True);a=p.parse_args()
    import onnx
    base=a.baseline.resolve();out=a.output.resolve();root=Path(__file__).resolve().parent
    if out.exists():raise RuntimeError('Fresh output directory required')
    before={str(f.relative_to(base)):sha(f) for f in base.rglob('*') if f.is_file()}
    shutil.copytree(base,out,copy_function=os.link)
    library=out/'libs/libpaired_projection.so'
    flags=['-shared','-fPIC','-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-fvisibility=hidden','-Wl,-Bsymbolic-functions']
    cmd=['g++',*flags,'-I'+str(a.ort_include),str(root/'paired_projection.cpp'),str(a.core),'-o',str(library)]
    subprocess.run(cmd,check=True)
    manifest=json.loads((out/'bundle.json').read_text());native=manifest['native']['Linux/x86_64']
    entry=manifest['streaming']['models'][native['model']];modelpath=out/entry['model']
    graph=onnx.load(modelpath,load_external_data=False)
    targets=[n for n in graph.graph.node if n.op_type=='PrecisionMatMulF32' and {v.name:onnx.helper.get_attribute_value(v) for v in n.attribute}.get('M')==8192]
    assert len(targets)==2
    current=next(n for n in targets if 'current' in n.name);previous=next(n for n in targets if 'previous' in n.name)
    assert current.input[1]==previous.input[1]
    for node in targets:
        attrs={v.name:onnx.helper.get_attribute_value(v) for v in node.attribute}
        assert attrs=={'K':2048,'M':8192,'backend':1,'native_abi':1,'precision_mode':8,'shards':1},attrs
    new=onnx.helper.make_node('PackedProjectionPairF32',[current.input[0],previous.input[0],current.input[1]],
        [current.output[0],previous.output[0]],name='stream_first_projection_pair',
        domain='fast.audiovae.streaming.matrix.experimental',native_abi=1,M=8192,K=2048)
    nodes=[];added=False
    for node in graph.graph.node:
        if node.name in {n.name for n in targets}:
            if not added:nodes.append(new);added=True
        else:nodes.append(node)
    del graph.graph.node[:];graph.graph.node.extend(nodes)
    graph.opset_import.append(onnx.helper.make_opsetid(new.domain,1))
    modelpath.unlink();onnx.save(graph,modelpath)
    entry['model_sha256']=sha(modelpath)
    entry['additional_libraries']=[*native['additional_libraries'],{'library':str(library.relative_to(out)),'sha256':sha(library)}]
    entry['projection_candidate']={'target':'first_pair_only','direct_frames':[1,2,3,4],'fallback':'accepted_INT8_core_for_larger_chunks','trained_tensors_unchanged':True}
    (out/'bundle.json').unlink();(out/'bundle.json').write_text(json.dumps(manifest,indent=2)+'\n')
    after={str(f.relative_to(base)):sha(f) for f in base.rglob('*') if f.is_file()}
    assert before==after,'Baseline changed'
    unchanged=onnx.load(base/entry['model'],load_external_data=False)
    assert [t.SerializeToString() for t in unchanged.graph.initializer]==[t.SerializeToString() for t in graph.graph.initializer]
    original_other=[n.SerializeToString() for n in unchanged.graph.node if n.name not in {x.name for x in targets}]
    actual_other=[n.SerializeToString() for n in graph.graph.node if n.name!=new.name]
    assert original_other==actual_other
    record={'baseline':str(base),'bundle':str(out),'source_sha256':{x.name:sha(x) for x in root.glob('*') if x.is_file()},
        'command':cmd,'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
        'ort_headers':{x.name:sha(x) for x in a.ort_include.glob('*.h')},'accepted_core_sha256':sha(a.core),
        'library_sha256':sha(library),'model_sha256':sha(modelpath),'baseline_unchanged':True,'initializers_unchanged':True,
        'other_nodes_unchanged':True,'original_nodes':[n.name for n in targets],
        'extra_packed_bytes':2*8192*2048,'gpu_used':False,'inference_run':False}
    (out/'projection-build.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
