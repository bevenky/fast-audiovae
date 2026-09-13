"""Build isolated paired scheduling experiment with the pinned CPU math code."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
REPO=ROOT/'work/fast-audiovae-streaming-baseline';SRC=REPO/'native/apple/streaming';OUT=HERE/'build'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'outputs/apple-streaming-promotion-v1/config-r3.json').read_text());bundle=Path(cfg['bundle'])
    man=json.loads((bundle/'bundle.json').read_text());pin=json.loads((SRC/'sources.json').read_text())
    dep=man['native']['Darwin/arm64']['dependencies'][0];dep_path=bundle/dep['library']
    assert sha(dep_path)==dep['sha256'];shutil.copy2(dep_path,OUT/dep_path.name)
    cap=lambda a:subprocess.check_output(a,text=True).strip()
    cc=cap(['xcrun','--find','clang']);cxx=cap(['xcrun','--find','clang++']);sdk=cap(['xcrun','--sdk','macosx','--show-sdk-path'])
    headers=REPO/'.deps/onnxruntime/include';xsmm=REPO/'.deps/apple-streaming/libxsmm-55a8fa6a1e479dec1f5ddbe20684c1cdc0ff7eb1/include'
    common=['-O3','-fPIC','-fvisibility=hidden','-fno-fast-math','-ffp-contract=off','-isysroot',sdk,'-I'+str(headers),'-I'+str(SRC/'kleidiai')]
    cpp=[*common,'-std=c++17','-isystem',str(Path(sdk)/'usr/include/c++/v1')]
    paths=[SRC/'multitile/native.cpp',SRC/'multitile/ort_ops.cpp',SRC/'libxsmm_panel/native.cpp',SRC/'libxsmm_panel/ort_ops.cpp',HERE/'paired_ops.cpp',HERE/'multitile_tasks.h']
    paths += [SRC/r for r in pin['kleidiai']['families']['multitile']]
    before={str(p):sha(p) for p in paths};commands=[]
    def run(a):
        a=list(map(str,a));commands.append(a);subprocess.run(a,check=True)
    objects=[]
    for i,r in enumerate(pin['kleidiai']['families']['multitile']):
        p=SRC/r;o=OUT/f'kai_{i}.o';extra=['-std=c11','-fno-vectorize','-fno-slp-vectorize'] if p.suffix=='.c' else []
        run([cc,*common,'-march=armv9.2-a+sme2',*extra,'-c',p,'-o',o]);objects.append(o)
    multi=OUT/'libdist_multitile_core.dylib';xs=OUT/'libdist_xsmm_core.dylib'
    run([cxx,*cpp,SRC/'multitile/native.cpp',*objects,'-dynamiclib','-Wl,-install_name,@rpath/'+multi.name,'-o',multi])
    run([cxx,*cpp,'-I'+str(xsmm),SRC/'libxsmm_panel/native.cpp',OUT/dep_path.name,'-dynamiclib','-Wl,-rpath,@loader_path','-Wl,-install_name,@rpath/'+xs.name,'-o',xs])
    for family,core,name in [('multitile',multi,'libdist_multitile_ort.dylib'),('libxsmm_panel',xs,'libdist_xsmm_ort.dylib')]:
        run([cxx,*cpp,SRC/family/'ort_ops.cpp',core,'-dynamiclib','-framework','Accelerate','-Wl,-rpath,@loader_path','-Wl,-install_name,@rpath/'+name,'-o',OUT/name])
    paired=OUT/'libdist_paired_ort.dylib'
    run([cxx,*cpp,HERE/'paired_ops.cpp',multi,xs,'-dynamiclib','-framework','Accelerate','-Wl,-rpath,@loader_path','-Wl,-install_name,@rpath/'+paired.name,'-o',paired])
    assert before=={str(p):sha(p) for p in paths},'Source changed during build'
    receipt={'sources':before,'commands':commands,'cpu_only':True,'graph':cfg['stream_graph_sha256'],'files':{str(p):sha(p) for p in OUT.glob('*.dylib')},'headers':{str(p):sha(p) for p in headers.glob('*.h')}}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Isolated paired CPU build complete.')
if __name__=='__main__':main()
