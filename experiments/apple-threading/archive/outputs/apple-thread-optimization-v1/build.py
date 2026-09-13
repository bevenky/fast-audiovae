"""Build only the two isolated scheduling candidates; reuse pinned math libraries."""
from pathlib import Path
import hashlib, json, shutil, subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPO = ROOT / 'work/fast-audiovae-streaming-baseline'
SRC = REPO / 'native/apple/streaming'
OUT = HERE / 'build'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT/'outputs/apple-streaming-promotion-v1/config-r3.json').read_text())
    bundle = Path(cfg['bundle'])
    manifest = json.loads((bundle/'bundle.json').read_text())
    record = manifest['native']['Darwin/arm64']['dependencies'][0]
    dep = bundle/record['library']
    assert sha(dep)==record['sha256']
    shutil.copy2(dep, OUT/dep.name)
    paths = ['state/custom_ops.cpp','libxsmm_panel/native.cpp','libxsmm_panel/ort_ops.cpp']
    hashes = {p:sha(SRC/p) for p in paths}
    cc = subprocess.check_output(['xcrun','--find','clang'],text=True).strip()
    cxx = subprocess.check_output(['xcrun','--find','clang++'],text=True).strip()
    sdk = subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip()
    headers = REPO/'.deps/onnxruntime/include'
    xsmm = REPO/'.deps/apple-streaming/libxsmm-55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1/include'
    common=['-O3','-fPIC','-fvisibility=hidden','-fno-fast-math','-ffp-contract=off',
            '-isysroot',sdk,'-I'+str(headers),'-I'+str(REPO/'native/apple')]
    cpp=[*common,'-std=c++17','-isystem',str(Path(sdk)/'usr/include/c++/v1')]
    commands=[]
    def run(args):
        commands.append(list(map(str,args)))
        subprocess.run(commands[-1],check=True)
    run([cc,*common,'-std=c11','-fno-vectorize','-fno-slp-vectorize','-DNCC_USE_ACCELERATE=1',
         '-c',REPO/'native/apple/native_kernels.c','-o',OUT/'base.o'])
    state=OUT/'libthread_state.dylib'
    run([cxx,*cpp,SRC/'state/custom_ops.cpp',OUT/'base.o','-dynamiclib','-framework','Accelerate',
         '-Wl,-install_name,@rpath/'+state.name,'-o',state])
    core=OUT/'libthread_xsmm_core.dylib'
    run([cxx,*cpp,'-I'+str(xsmm),SRC/'libxsmm_panel/native.cpp',OUT/dep.name,'-dynamiclib',
         '-Wl,-rpath,@loader_path','-Wl,-install_name,@rpath/'+core.name,'-o',core])
    bridge=OUT/'libthread_xsmm_ort.dylib'
    run([cxx,*cpp,SRC/'libxsmm_panel/ort_ops.cpp',core,'-dynamiclib','-framework','Accelerate',
         '-Wl,-rpath,@loader_path','-Wl,-install_name,@rpath/'+bridge.name,'-o',bridge])
    assert hashes=={p:sha(SRC/p) for p in paths}, 'Source changed during compile'
    dependencies={str(p):sha(p) for p in [REPO/'native/apple/native_kernels.c',REPO/'native/apple/native_kernels.h',
        *sorted(headers.glob('*.h')),*sorted(xsmm.glob('*.h'))]}
    (OUT/'receipt.json').write_text(json.dumps(dict(sources=hashes,commands=commands,
        compile_dependencies=dependencies,
        files={str(p):sha(p) for p in [state,core,bridge,OUT/dep.name]},
        graph=cfg['stream_graph_sha256'],cpu_only=True),indent=2)+'\n')
    print('Built isolated state and matrix candidates; production artifacts unchanged.')
if __name__=='__main__': main()
