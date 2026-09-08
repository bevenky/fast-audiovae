"""Record and supply transitive quoted includes from a hash-verified MKL include wheel."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--pins',type=Path,required=True);p.add_argument('--dependency',type=Path,required=True);a=p.parse_args()
    pins=json.loads(a.pins.read_text());package=next(x for x in pins['packages'] if x['filename'].startswith('mkl_include-'))
    wheel=a.dependency/package['filename'];assert sha(wheel)==package['sha256']
    out=a.dependency/'include';seen={};pending=list(pins['header_sha256'])
    with zipfile.ZipFile(wheel) as z:
        names={}
        for n in z.namelist():
            if n.endswith('.h'):names.setdefault(Path(n).name,[]).append(n)
        while pending:
            name=pending.pop()
            if name in seen:continue
            entries=names.get(name,[])
            if len(entries)!=1:raise ValueError('Missing or duplicate header '+name)
            data=z.read(entries[0]);digest=hashlib.sha256(data).hexdigest();target=out/name
            if target.exists():assert sha(target)==digest
            else:target.write_bytes(data)
            seen[name]=digest
            pending.extend(re.findall(r'^\s*#\s*include\s*"([^"]+)"',data.decode(),flags=re.M))
    assert all(seen[k]==v for k,v in pins['header_sha256'].items())
    record={'include_wheel_sha256':sha(wheel),'header_sha256':seen,'method':'Recursive quoted includes beginning at the four frozen public headers; full official wheel digest verified before extraction.','gpu_runtime_installed':False,'script_sha256':sha(Path(__file__))}
    (a.dependency/'header-closure.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
