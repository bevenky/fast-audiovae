"""Build only the new second-pair scheduling wrappers and pinned sweep core."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
REPO=ROOT/'work/fast-audiovae-streaming-baseline';SRC=REPO/'native/apple/streaming';OUT=HERE/'build'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'outputs/apple-streaming-promotion-v1/config-r3.json').read_text());pin=json.loads((SRC/'sources.json').read_text())
    cap=lambda a:subprocess.check_output(a,text=True).strip()
    cc=cap(['xcrun','--find','clang']);cxx=cap(['xcrun','--find','clang++']);sdk=cap(['xcrun','--sdk','macosx','--show-sdk-path'])
    headers=REPO/'.deps/onnxruntime/include'
    common=['-O3','-fPIC','-fvisibility=hidden','-fno-fast-math','-ffp-contract=off','-isysroot',sdk,'-I'+str(headers),'-I'+str(SRC/'kleidiai')]
    cpp=[*common,'-std=c++17','-isystem',str(Path(sdk)/'usr/include/c++/v1')]
    paths=[SRC/'sweep/native.cpp',SRC/'sweep/ort_ops.cpp',SRC/'sweep/geometry.h',HERE/'paired_ops.cpp',HERE/'sweep_tasks.h']
    paths += [SRC/r for r in pin['kleidiai']['families']['sweep']]
    # Include all shipped KAI headers used transitively by the pinned sources.
    paths += list((SRC/'kleidiai').rglob('*.h'))+list(headers.glob('*.h'))
    before={str(p):sha(p) for p in paths};commands=[]
    def run(a):
        a=list(map(str,a));commands.append(a);subprocess.run(a,check=True)
    objects=[]
    for i,r in enumerate(pin['kleidiai']['families']['sweep']):
        p=SRC/r;o=OUT/f'kai_{i}.o';extra=['-std=c11','-fno-vectorize','-fno-slp-vectorize'] if p.suffix=='.c' else []
        run([cc,*common,'-march=armv9.2-a+sme2',*extra,'-c',p,'-o',o]);objects.append(o)
    core=OUT/'libdist3_sweep_core.dylib'
    run([cxx,*cpp,SRC/'sweep/native.cpp',*objects,'-dynamiclib','-Wl,-install_name,@rpath/'+core.name,'-o',core])
    for source,name in [(SRC/'sweep/ort_ops.cpp','libdist3_sweep_ort.dylib'),(HERE/'paired_ops.cpp','libdist3_paired_ort.dylib')]:
        run([cxx,*cpp,source,core,'-dynamiclib','-framework','Accelerate','-Wl,-rpath,@loader_path','-Wl,-install_name,@rpath/'+name,'-o',OUT/name])
    assert before=={str(p):sha(p) for p in paths},'Source changed during build'
    receipt={'sources':before,'commands':commands,'cpu_only':True,'graph':cfg['stream_graph_sha256'],'files':{str(p):sha(p) for p in OUT.glob('*.dylib')}}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Isolated second-pair CPU build complete.')
if __name__=='__main__':main()
