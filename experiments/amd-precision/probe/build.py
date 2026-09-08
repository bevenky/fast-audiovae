"""Build optional CPU-only attribution wrappers, never used for headline timing."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser()
    for name in ('precision-header','mkl-include','aocl-include','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve()
    if out.exists():raise ValueError('Fresh probe build required')
    assert sha(a.precision_header)=='c8758086e9346c1522802d86ea9be59a067fe0bc5e9702560c435b0960724ac0'
    paths=[ROOT/'precision_probe.cpp',a.precision_header,*a.mkl_include.rglob('*.h'),*a.aocl_include.rglob('*.h')]
    before={str(p.resolve()):sha(p) for p in paths};out.mkdir(parents=True);commands=[]
    for backend,include,flags in [('mkl',a.mkl_include,[]),('aocl',a.aocl_include,['-DPROBE_AOCL=1'])]:
        commands.append(['g++','-shared','-fPIC','-O2','-std=c++17','-fno-fast-math','-ffp-contract=off','-fvisibility=hidden',*flags,'-I'+str(a.precision_header.resolve().parent),'-I'+str(include.resolve()),str(ROOT/'precision_probe.cpp'),'-Wl,-z,defs','-ldl','-pthread','-o',str(out/('libprecision_probe_'+backend+'.so'))])
    for command in commands:subprocess.run(command,check=True)
    assert before=={str(p.resolve()):sha(p) for p in paths}
    r={'scope':'CPU-only optional call attribution; no model run or headline timing','sources_and_headers':before,'commands':commands,'libraries':{str(p):sha(p) for p in out.glob('*.so')},'script_sha256':sha(Path(__file__))}
    (out/'build.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['libraries'],indent=2))
if __name__=='__main__':main()
