"""Freeze the Apple port's four-model comparison without executing a decoder."""
import argparse,copy,hashlib,json,os
from pathlib import Path
import onnx
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--graph-manifest',type=Path,required=True);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh config required')
    source=json.loads(a.source_config.read_text());graph=json.loads(a.graph_manifest.read_text());build=json.loads(a.build.read_text())
    assert graph['changed_product_count']==22 and graph['backend']==2 and graph['shards']==4
    assert sha(graph['path'])==graph['sha256']
    for name,digest in build['libraries'].items():assert sha(a.build.parent/name)==digest
    cfg={key:copy.deepcopy(source[key]) for key in ['expected_case_count','expected_uids','timing_uids','timing_selection','frozen_manifest_sha256']}
    cfg['kind_contracts']={'audio':{'sample_rate':48000,'hop':1920,'latent_channels':64},
                           'mimi':{'sample_rate':24000,'hop':1920,'latent_channels':32}}
    cfg['models']=[];cfg['artifact_sha256']={};base=a.output.resolve().parent
    def pin(path):
        path=Path(path).resolve();rel=os.path.relpath(path,base);cfg['artifact_sha256'][rel]=sha(path);return rel
    originals={m['name']:m for m in source['models']}
    for name,source_name in [('audio_stock','audio_stock'),('fast_fp32','fast_default'),('int8_large','fast_default'),('mimi','mimi')]:
        model=copy.deepcopy(originals[source_name]);model['name']=name;model['causal']=True
        if name=='int8_large':
            model['path']=graph['path'];model['approximate']=True
            model['custom_libraries']=[model.pop('custom_library'),build['ops_path']]
        absolute_model=Path(model['path']).resolve();m=onnx.load(absolute_model,load_external_data=False)
        locations=sorted({str(absolute_model.parent/entry.value) for tensor in m.graph.initializer
                          for entry in tensor.external_data if entry.key=='location'})
        model['path']=pin(absolute_model)
        if locations:model['external_data']=[pin(x) for x in locations]
        if model.get('custom_library'):model['custom_library']=pin(model['custom_library'])
        if model.get('custom_libraries'):model['custom_libraries']=[pin(x) for x in model['custom_libraries']]
        cfg['models'].append(model)
    for kind in ('audio','mimi'):
        cfg[kind+'_cases']=pin(source[kind+'_cases']);pin(Path(source[kind+'_cases']).with_suffix('.json'))
    cfg['transitive_core_library']=pin(build['core_path'])
    cfg['port_provenance']={'source_config':pin(a.source_config),'graph_manifest':pin(a.graph_manifest),'build':pin(a.build),
                            'scope':'Initial Apple CPU screen before full quality/speed validation; no default change'}
    base.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(cfg,indent=2)+'\n');print(sha(a.output))
if __name__=='__main__':main()
