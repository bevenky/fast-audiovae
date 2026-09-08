"""Fetch and extract only hash-pinned CPU libraries/headers into a new local directory."""
import argparse,hashlib,json,os,shutil,urllib.request,zipfile
from pathlib import Path

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pins',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    out=a.output_dir.resolve()
    if out.exists():raise ValueError('Fresh dependency directory required')
    pins=json.loads(a.pins.read_text());assert pins['version']=='2026.1.0'
    out.mkdir(parents=True);(out/'include').mkdir();(out/'lib').mkdir()
    records=[]
    for package in pins['packages']:
        wheel=out/package['filename']
        with urllib.request.urlopen(package['url']) as source,wheel.open('xb') as target:shutil.copyfileobj(source,target)
        assert sha(wheel)==package['sha256'] and wheel.stat().st_size==package['wheel_bytes']
        wanted=pins['header_sha256'] if package['filename'].startswith('mkl_include-') else pins['cpu_library_sha256']
        dest=out/('include' if package['filename'].startswith('mkl_include-') else 'lib')
        with zipfile.ZipFile(wheel) as archive:
            for name,digest in wanted.items():
                entries=[x for x in archive.namelist() if Path(x).name==name]
                if len(entries)!=1:raise ValueError('Missing or duplicate pinned entry '+name)
                target=dest/name
                with archive.open(entries[0]) as source,target.open('xb') as stream:shutil.copyfileobj(source,stream)
                if sha(target)!=digest:raise ValueError('Entry hash mismatch '+name)
        records.append({'filename':package['filename'],'sha256':sha(wheel),'extracted_files':list(wanted)})
    record={'status':'provisioned_not_executed','cpu_libraries_only':True,'gpu_runtime_installed':False,'version':pins['version'],'pins_sha256':sha(a.pins),'script_sha256':sha(Path(__file__)),'packages':records}
    (out/'provision.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
