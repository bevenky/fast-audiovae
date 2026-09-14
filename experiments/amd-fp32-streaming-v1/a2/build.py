"""Build only on Linux x86; no inference or metadata probe is executed."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser()
    for name in ('repo','baseline-build','sleef-prefix','ort-include','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();repo=a.repo.resolve();out=a.output.resolve()
    assert platform.system()=='Linux' and platform.machine() in ('x86_64','amd64'),'Linux x86 only'
    out.mkdir(parents=True,exist_ok=False)
    report={'complete':False,'scope':'Build only, no operator execution','commands':[]}
    try:
        pins=load(HERE/'source-pins.json')
        for rel,want in pins.items():assert sha(repo/rel)==want,('Source changed',rel)
        spec=importlib.util.spec_from_file_location('a2_x86_builder',repo/'tools/build_x86.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        mod.require_headers(a.ort_include.resolve())
        baseline=load(a.baseline_build);fp=baseline['fingerprint']
        baseline_lib=Path(baseline['library']).resolve()
        assert sha(baseline_lib)==baseline['library_sha256']
        for rel in ('native/x86/native_kernels.c','native/x86/native_kernels.h','native/x86/custom_ops.cpp'):
            assert fp['source_sha256'][rel]==pins[rel],('Control source differs',rel)
        prefix=a.sleef_prefix.resolve();archives=[x for x in (prefix/'lib/libsleef.a',prefix/'lib64/libsleef.a') if x.is_file()]
        assert len(archives)==1
        archive=archives[0];assert sha(archive)==fp['sleef_library_sha256']
        assert sha(prefix/'include/sleef.h')==fp['sleef_header_sha256']
        cc,cxx,versions=mod.compilers()
        novec=['-fno-vectorize','-fno-slp-vectorize'] if 'clang' in versions['cc'].lower() else ['-fno-tree-vectorize']
        assert fp['compilers']==versions and fp['common_flags']==mod.COMMON_FLAGS and fp['c_novec_flags']==novec
        assert fp['openmp'] is False and fp['tile']==256 and fp['explicit_avx512_backend']==5
        inputs={str(x.resolve()):sha(x) for x in [HERE/'direct_history.c',HERE/'custom_ops.cpp',HERE/'build.py',HERE/'source-pins.json',a.baseline_build,baseline_lib,archive,prefix/'include/sleef.h']}
        for name in mod.HEADERS:inputs[str((a.ort_include/name).resolve())]=sha(a.ort_include/name)
        for rel in pins:inputs[str(repo/rel)]=sha(repo/rel)
        (out/'native_wrapper.c').write_text('#include "native_kernels.c"\n#include "direct_history.c"\n')
        library=out/'libaudiovae_amd_a2_rawhistory.so'
        commands=[
            [cc,*mod.COMMON_FLAGS,*novec,'-std=c11','-DNCC_USE_SLEEF=1','-I'+str(prefix/'include'),'-I'+str(repo/'native/x86'),'-I'+str(HERE),'-c',str(out/'native_wrapper.c'),'-o',str(out/'native.o')],
            [cxx,*mod.COMMON_FLAGS,'-std=c++17','-I'+str(a.ort_include.resolve()),'-I'+str(repo/'native/x86'),'-c',str(HERE/'custom_ops.cpp'),'-o',str(out/'bridge.o')],
            [cxx,'-shared','-pthread',str(out/'native.o'),str(out/'bridge.o'),str(archive),'-lm','-Wl,-Bsymbolic','-Wl,--exclude-libs,ALL','-Wl,-z,defs','-Wl,-soname,'+library.name,'-o',str(library)]]
        report.update(inputs=inputs,baseline_build_sha256=sha(a.baseline_build),baseline_library=str(baseline_lib),
            baseline_library_sha256=sha(baseline_lib),compiler=versions,flags=mod.COMMON_FLAGS,c_novec_flags=novec,
            domain='fast.audiovae.amd.rawhistory.a2.v1',backend=5,math='same static SLEEF as control',tile=256,openmp=False)
        for command in commands:
            r=subprocess.run(command,capture_output=True,text=True,env=mod.cpu_environment())
            report['commands'].append({'argv':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
            assert r.returncode==0,'Compiler/linker failed; see build.json'
        assert all(sha(k)==v for k,v in inputs.items()),'Build inputs changed'
        report.update(complete=True,library=str(library),library_sha256=sha(library),wrapper_sha256=sha(out/'native_wrapper.c'))
    except BaseException as e:
        report['error']=repr(e);raise
    finally:(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
