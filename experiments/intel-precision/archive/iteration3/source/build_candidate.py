"""Build an isolated Intel CPU candidate with namespaced operators and pinned dependencies."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = Path('/var/tmp/fast-audiovae-intel-precision')
REPO = Path('/var/tmp/fast-audiovae-intel-iteration2/repo')
ORT = Path('/var/tmp/fast-audiovae-20260907/repo/.deps/onnxruntime/include')
MKL = Path('/var/tmp/fast-audiovae-20260907/mkl_candidate/dependency')
NATIVE = Path('/var/tmp/fast-audiovae-fusion-20260907/repo/.build/x86/libfast_audiovae_x86_7312a0b7f908.so')
XSMM = Path('/var/tmp/fast-audiovae-fusion-20260907/dependency/libxsmm-55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1')
SLEEF = Path('/var/tmp/fast-audiovae-20260907/repo/.deps/sleef')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Intel Linux build required')
    out = args.output_dir.resolve()
    require(not out.exists(), 'Use a fresh output directory')
    require(sha(NATIVE) == '168f57d2e6049bcffc64d61ad0722d2f372b037ad3aabfd2b5cedd963292082a', 'Native dependency changed')
    require(sha(SLEEF/'lib/libsleef.a') == 'a3e4dee0f28628339d8c98745a71afe8ab0dcb898b789f8b42a050e69986f8f5', 'SLEEF archive changed')
    require(sha(SLEEF/'include/sleef.h') == '2f8add3a70ea0653d15dd29df4ec999ca33fd995cda9fe283ab01951df37623a', 'SLEEF header changed')
    mkl_pins = json.loads((EXPERIMENT/'pins/mkl.json').read_text())
    ort_pins = json.loads((REPO/'experiments/cpu-stage/dependency-pins.json').read_text())['ort_headers']['sha256']
    for name, digest in mkl_pins['header_sha256'].items():
        require(sha(MKL/'include'/name) == digest, 'MKL header changed: '+name)
    for name, digest in mkl_pins['cpu_library_sha256'].items():
        require(sha(MKL/'lib'/name) == digest, 'MKL library changed: '+name)
    for name, digest in ort_pins.items():
        require(sha(ORT/name) == digest, 'ORT header changed: '+name)
    require('#define ORT_API_VERSION 29' in (ORT/'onnxruntime_c_api.h').read_text(), 'ORT API29 required')
    sources = [p for d in ('core', 'pipeline') for p in (ROOT/d).rglob('*') if p.suffix in ('.h','.c','.cpp','.py')]
    dependencies = [NATIVE,SLEEF/'lib/libsleef.a',SLEEF/'include/sleef.h',XSMM/'lib/libxsmm.a']
    dependencies += [ORT/name for name in ort_pins]
    dependencies += [MKL/'include'/name for name in mkl_pins['header_sha256']]
    dependencies += [MKL/'lib'/name for name in mkl_pins['cpu_library_sha256']]
    dependencies += [REPO/'experiments/cpu-stage'/name for name in ('matrix/fused_pointwise.c','matrix/fused_pointwise.h','upsample/projection.c','upsample/projection.h')]
    dependencies += [REPO/'native/x86/native_kernels.h',*[p for p in (XSMM/'include').rglob('*.h')]]
    inputs = sources+dependencies+[Path(__file__).resolve()]
    before = {str(p):sha(p) for p in inputs}
    out.mkdir(parents=True)
    flags = ['-O3','-fPIC','-fvisibility=hidden','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror']
    defines = ['-DIP_ITERATION3_NAMESPACE=1']
    includes = [ORT,ROOT/'core',ROOT/'pipeline',REPO/'native/x86',REPO/'experiments/cpu-stage/matrix',REPO/'experiments/cpu-stage/upsample',SLEEF/'include']
    inc = ['-I'+str(p) for p in includes]
    link_flags = ['-Wl,-Bsymbolic-functions','-Wl,-z,defs','-Wl,--exclude-libs,ALL','-Wl,--disable-new-dtags','-Wl,-rpath,'+str(out),'-Wl,-rpath,'+str(NATIVE.parent),'-Wl,-rpath,'+str(MKL/'lib'),'-ldl','-pthread','-lm']
    core = out/'libintel_precision3_core.so'
    commands = [['g++',*flags,'-std=c++17',*defines,'-DIP_WITH_MKL=1','-I'+str(MKL/'include'),'-shared',str(ROOT/'core/precision.cpp'),'-Wl,--no-as-needed',*[str(MKL/'lib'/n) for n in mkl_pins['direct_link_libraries']],*link_flags,'-Wl,-soname,'+core.name,'-o',str(core)]]
    commands.append(['g++',*flags,'-std=c++17',*defines,*inc,'-shared',str(ROOT/'core/custom_op.cpp'),str(core),*link_flags,'-o',str(out/'libintel_precision3_ops.so')])
    for name, source, define in [('matrix',REPO/'experiments/cpu-stage/matrix/fused_pointwise.c','FX_WITH_LIBXSMM'),('projection',REPO/'experiments/cpu-stage/upsample/projection.c','UP_WITH_LIBXSMM')]:
        commands.append(['gcc',*flags,'-std=c11','-D'+define+'=1','-I'+str(XSMM/'include'),'-c',str(source),'-o',str(out/(name+'.o'))])
    for name in ('stage_precision','upsample_precision','stage_fp32'):
        commands.append(['g++',*flags,'-std=c++17',*defines,*inc,'-c',str(ROOT/'pipeline'/(name+'.cpp')),'-o',str(out/(name+'.o'))])
        objects=[out/(name+'.o'),out/'matrix.o']+([out/'projection.o'] if name=='upsample_precision' else [])
        commands.append(['g++','-shared',*[str(p) for p in objects],str(core),str(NATIVE),str(XSMM/'lib/libxsmm.a'),str(SLEEF/'lib/libsleef.a'),*link_flags,'-o',str(out/('libiteration3_'+name+'.so'))])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1',NVIDIA_VISIBLE_DEVICES='void',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',BLIS_NUM_THREADS='1')
    for key in ('LD_PRELOAD','MKL_CBWR','MKL_ENABLE_INSTRUCTIONS','LIBXSMM_TARGET'):
        env.pop(key,None)
    for command in commands:
        subprocess.run(command,check=True,env=env)
    require({str(p):sha(p) for p in inputs} == before, 'Build input changed')
    libraries = {str(p):sha(p) for p in out.glob('*.so')}
    exports = subprocess.check_output(['nm','-D','--defined-only',str(core)],text=True)
    require(' ip3_create' in exports and ' ip_create' not in exports, 'Candidate core symbol isolation failed')
    elf={}
    for library in libraries:
        defined=subprocess.check_output(['nm','-D','--defined-only',library],text=True)
        undefined=subprocess.check_output(['nm','-D','--undefined-only',library],text=True)
        dynamic=subprocess.check_output(['readelf','-d',library],text=True)
        require(not any(line.split()[-1].startswith(('ip_','Sleef_','fx_')) for line in defined.splitlines() if line.split()),'Candidate exports an unisolated dependency symbol')
        require(not any(line.split()[-1].startswith('ip_') for line in undefined.splitlines() if line.split()),'Candidate references baseline precision symbols')
        elf[library]={'defined_symbols':defined,'undefined_symbols':undefined,'dynamic':dynamic}
    record={'status':'built','GPU_used':False,'ORT_API':29,'cpu_kernel_execution':False,'commands':commands,'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],'input_sha256':before,'libraries':libraries,'core_exports':exports,'elf':elf,'base_native_sha256':sha(NATIVE),'sleef_version':'3.9.0','sleef_archive_sha256':sha(SLEEF/'lib/libsleef.a')}
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'built','libraries':libraries}))


if __name__ == '__main__':
    main()
