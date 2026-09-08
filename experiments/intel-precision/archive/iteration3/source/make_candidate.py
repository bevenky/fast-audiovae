"""Namespace the unchanged selective INT8 graph for the new CPU kernels."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-config',required=True,type=Path)
    parser.add_argument('--baseline-config-sha256',required=True)
    parser.add_argument('--build',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--tile-policy',choices=('unchanged','upsample256','c256_128','both'),default='unchanged')
    a=parser.parse_args()
    if sha(a.baseline_config)!=a.baseline_config_sha256:raise ValueError('Baseline config changed')
    import onnx
    config=json.loads(a.baseline_config.read_text())
    source_root=a.baseline_config.resolve().parent
    def resolve(p):return (source_root/p).resolve()
    baseline=next(m for m in config['models'] if m['name']=='int8_large')
    original=resolve(baseline['path'])
    if sha(original)!=config['artifact_sha256'][baseline['path']]:raise ValueError('Baseline graph changed')
    model=onnx.load(original)
    initializers=[v.SerializeToString() for v in model.graph.initializer]
    domains={'fast.audiovae.precision.matrix.experimental':'fast.audiovae.precision.iteration3',
             'fast.audiovae.precision.stage.experimental':'fast.audiovae.precision.stage.iteration3',
             'fast.audiovae.precision.upsample.experimental':'fast.audiovae.precision.upsample.iteration3',
             'fast.audiovae.stage.experimental':'fast.audiovae.stage.iteration3'}
    changed={k:0 for k in domains};tiles=[]
    for node in model.graph.node:
        if node.domain in domains:
            old=node.domain;node.domain=domains[old];changed[old]+=1
            attrs={v.name:v for v in node.attribute}
            if old=='fast.audiovae.precision.upsample.experimental' and a.tile_policy in ('upsample256','both'):
                if attrs['tile_time'].i!=128:raise ValueError('Unexpected original upsample tile')
                attrs['tile_time'].i=256;tiles.append({'node':node.name,'from':128,'to':256})
            if old=='fast.audiovae.precision.stage.experimental' and a.tile_policy in ('c256_128','both'):
                if attrs['channels'].i!=256 or attrs['tile_time'].i!=256:raise ValueError('Unexpected original C256 tile')
                attrs['tile_time'].i=128;tiles.append({'node':node.name,'from':256,'to':128})
    if list(changed.values())!=[14,1,1,2]:raise ValueError('Unexpected selective graph operator counts: '+str(changed))
    for domain in domains.values():model.opset_import.append(onnx.helper.make_opsetid(domain,1))
    if [v.SerializeToString() for v in model.graph.initializer]!=initializers:raise ValueError('Initializer changed')
    onnx.checker.check_model(model)
    out=a.output_dir.resolve()
    if out.exists():raise ValueError('Use a fresh graph/config output directory')
    out.mkdir(parents=True)
    graph=out/'decoder.onnx';onnx.save(model,graph)
    loaded=onnx.load(graph)
    if [v.SerializeToString() for v in loaded.graph.initializer]!=initializers:raise ValueError('Saved initializer changed')
    build=json.loads(a.build.read_text());libdir=a.build.resolve().parent
    for path,digest in build['libraries'].items():
        if sha(path)!=digest:raise ValueError('Candidate build library changed')
    candidate=copy.deepcopy(baseline);candidate['name']='int8_iteration3_'+a.tile_policy;candidate['path']=str(graph)
    # Only retained original operators plus newly namespaced operators are needed.
    candidate['custom_libraries']=[str(resolve(p)) for p in baseline['custom_libraries'][:4]]
    candidate['custom_libraries'] += [str(libdir/n) for n in ('libintel_precision3_ops.so','libiteration3_stage_precision.so','libiteration3_upsample_precision.so','libiteration3_stage_fp32.so')]
    candidate['external_data']=[]
    for model_config in config['models']:
        model_config['path']=str(resolve(model_config['path']))
        model_config['custom_libraries']=[str(resolve(p)) for p in model_config.get('custom_libraries',[])]
        model_config['external_data']=[str(resolve(p)) for p in model_config.get('external_data',[])]
    for k in ('audio_cases','mimi_cases'):config[k]=str(resolve(config[k]))
    config['artifact_sha256']={str(resolve(p)):h for p,h in config['artifact_sha256'].items()}
    config['artifact_sha256'].update(build['libraries'])
    config['artifact_sha256'][str(graph)]=sha(graph)
    config['models'].append(candidate)
    # The public harness requires paths relative to the configuration file.
    for model_config in config['models']:
        model_config['path']=os.path.relpath(model_config['path'],out)
        model_config['custom_libraries']=[os.path.relpath(p,out) for p in model_config.get('custom_libraries',[])]
        model_config['external_data']=[os.path.relpath(p,out) for p in model_config.get('external_data',[])]
    for key in ('audio_cases','mimi_cases'):config[key]=os.path.relpath(config[key],out)
    config['artifact_sha256']={os.path.relpath(p,out):h for p,h in config['artifact_sha256'].items()}
    config.pop('recovery',None)
    config['iteration3']={'source_config_sha256':a.baseline_config_sha256,'source_graph_sha256':sha(original),'build_sha256':sha(a.build),'unchanged_initializer_count':len(initializers),'domain_counts':changed,'tile_policy':a.tile_policy,'tile_changes':tiles,'new_quantization':False,'GPU_used':False,'script_sha256':sha(__file__)}
    destination=out/'candidate.json';destination.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'config':str(destination),'config_sha256':sha(destination),'model':candidate['name'],'graph_sha256':sha(graph),'changes':config['iteration3']}))


if __name__=='__main__':main()
