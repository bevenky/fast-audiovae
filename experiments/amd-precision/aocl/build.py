"""Build the isolated AOCL integer core and relink unchanged accepted fused objects."""
import argparse,hashlib,json,os,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LIB_SHA='6fb0e10f2074367c4b038beb14647b8c02cc67116f973d10b4a02f9ca01974b0'
COMMIT='c577191304a3db0029f2f12fcacbc8ad296a645d'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('aocl-root','ort-include','native-build','original-fused-build','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cxx',default='g++');a=p.parse_args();out=a.output_dir.resolve()
    if out.exists():raise ValueError('Fresh build directory required')
    assert platform.system()=='Linux' and platform.machine()=='x86_64'
    aocl=a.aocl_root.resolve();dependency=json.loads((aocl/'build.json').read_text())
    assert dependency['source_pin']['commit']==COMMIT
    lib=aocl/'install/lib/libaocl-dlp.so';assert sha(lib)==LIB_SHA
    inc=aocl/'install/include';include=a.ort_include.resolve()
    native=json.loads(a.native_build.read_text());nlib=Path(native['library']);assert sha(nlib)==native['library_sha256']
    old=json.loads(a.original_fused_build.read_text());obase=a.original_fused_build.resolve().parent
    source=ROOT/'source';derivation=json.loads((source/'derivation.json').read_text())
    for name,digest in derivation['sources'].items():assert sha(source/name)==digest
    for path,digest in old['sha256'].items():assert sha(Path(path))==digest
    assert '#define ORT_API_VERSION 29' in (include/'onnxruntime_c_api.h').read_text()
    xsmm=next(Path(x) for cmd in old['commands'] for x in cmd if x.endswith('/lib/libxsmm.a'))
    objects=[obase/(x+'.o') for x in ('stage','upsample','matrix','projection')]
    inputs=[lib,nlib,xsmm,*objects,*sorted(inc.rglob('*.h')),*sorted(include.glob('*.h')),*[source/x for x in derivation['sources']]]
    before={str(x):sha(x) for x in inputs}
    for header in (inc/'classic').glob('*.h'):
        upstream=aocl/'source/include/classic'/header.name
        assert upstream.is_file() and sha(header)==sha(upstream)
    out.mkdir(parents=True)
    core=out/'libamd_aocl_precision_core.so';ops=out/'libamd_aocl_precision_ops.so'
    stage=out/'libamd_aocl_precision_stage.so';up=out/'libamd_aocl_precision_upsample.so'
    flags=['-shared','-fPIC','-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-fvisibility=hidden']
    commands=[[a.cxx,*flags,'-DIP_WITH_AOCL=1','-I'+str(inc),str(source/'precision.cpp'),str(lib),'-Wl,-rpath,'+str(lib.parent),'-Wl,-soname,'+core.name,'-Wl,-Bsymbolic-functions','-lpthread','-lm','-ldl','-o',str(core)],
              [a.cxx,*flags,'-I'+str(include),str(source/'custom_op.cpp'),str(core),'-Wl,-rpath,$ORIGIN','-Wl,-Bsymbolic-functions','-o',str(ops)]]
    for name,target in [('stage',stage),('upsample',up)]:
        objs=[obase/(name+'.o'),obase/'matrix.o']+([obase/'projection.o'] if name=='upsample' else [])
        commands.append([a.cxx,'-shared',*map(str,objs),str(core),str(nlib),str(xsmm),'-Wl,-rpath,$ORIGIN','-Wl,-rpath,'+str(nlib.parent),'-ldl','-pthread','-lm','-o',str(target)])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',ROCR_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for command in commands:subprocess.run(command,check=True,env=env)
    assert before=={str(x):sha(x) for x in inputs}
    record={'status':'built_not_executed','gpu_used':False,'backend':'AOCL-DLP U8xS8->S32, one TLS inner worker','commands':commands,'source_derivation_sha256':sha(source/'derivation.json'),'compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],
        'sources':derivation['sources'],'core_path':str(core),'ops_path':str(ops),'libraries':{str(x):sha(x) for x in (core,ops,stage,up)},'sha256':before,'aocl_commit':COMMIT,'aocl_library_sha256':LIB_SHA,'fused_original_build_sha256':sha(a.original_fused_build),'script_sha256':sha(Path(__file__))}
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':record['status'],'libraries':record['libraries']},indent=2))
if __name__=='__main__':main()
