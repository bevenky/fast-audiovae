"""Swap only the integer-core dependencies in a frozen AMD portability config."""
import argparse,copy,hashlib,json,os
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser()
    for name in ('source-config','build','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-config-sha256',required=True);a=p.parse_args()
    assert sha(a.source_config)==a.source_config_sha256
    if a.output.exists():raise ValueError('Fresh config required')
    cfg=json.loads(a.source_config.read_text());build=json.loads(a.build.read_text());base=a.output.resolve().parent
    resolve=lambda x:(a.source_config.resolve().parent/x).resolve()
    rel=lambda x:os.path.relpath(Path(x).resolve(),base)
    oldpins={str(resolve(x)):v for x,v in cfg['artifact_sha256'].items()}
    files={Path(x).name:Path(x) for x in build['libraries']}
    replacements={'libintel_precision_ops.so':files['libamd_aocl_precision_ops.so'],
                  'libprecision_stage.so':files['libamd_aocl_precision_stage.so'],
                  'libprecision_upsample.so':files['libamd_aocl_precision_upsample.so']}
    pins={}
    def add(path,expected):
        path=Path(path).resolve();actual=sha(path)
        if actual!=expected:raise ValueError('Artifact hash mismatch '+str(path))
        pins[rel(path)]=actual;return rel(path)
    def old(x):
        path=resolve(x);return add(path,oldpins[str(path)])
    changed=[]
    for model in cfg['models']:
        model['path']=old(model['path']);model['external_data']=[old(x) for x in model.get('external_data',[])]
        if model.get('custom_library'):model['custom_library']=old(model['custom_library'])
        libraries=[]
        for value in model.get('custom_libraries',[]):
            path=resolve(value)
            if model['name']=='int8_large' and path.name in replacements:
                new=replacements[path.name];libraries.append(add(new,build['libraries'][str(new)]));changed.append(path.name)
            else:libraries.append(old(value))
        if libraries:model['custom_libraries']=libraries
    assert sorted(changed)==sorted(replacements)
    for kind in ('audio','mimi'):
        value=cfg[kind+'_cases'];cfg[kind+'_cases']=old(value);old(str(Path(value).with_suffix('.json')))
    for path,digest in build['libraries'].items():add(Path(path),digest)
    aocl=[(Path(path),digest) for path,digest in build['sha256'].items() if Path(path).name=='libaocl-dlp.so']
    assert len(aocl)==1;add(*aocl[0])
    cfg['artifact_sha256']=pins;cfg['source_config_sha256']=a.source_config_sha256
    cfg['scope']='AMD CPU-only selective INT8 using pinned AOCL-DLP integer GEMM. Same graph, scales, weights, sharding and nonlinear math as initial portability screen; only integer backend changes.'
    cfg['aocl_derivation']={'script_sha256':sha(Path(__file__)),'build_sha256':sha(a.build),'commit':build['aocl_commit'],'library_sha256':build['aocl_library_sha256'],'inner_threads':1,'outer_shards_segments':2,'quality_status':'pending'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps({'config':str(a.output),'sha256':sha(a.output),'artifacts':len(pins)},indent=2))
if __name__=='__main__':main()
