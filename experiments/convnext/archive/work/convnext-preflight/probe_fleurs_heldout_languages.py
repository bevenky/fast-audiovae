import json
import urllib.request
import urllib.parse
from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq

api,fs = HfApi(),HfFileSystem()
for language in ['cmn_hans_cn','ar_eg','de_de']:
    for kind,url in [('parquet',f'https://huggingface.co/api/datasets/google/fleurs/parquet/{language}/test'),('rows','https://datasets-server.huggingface.co/first-rows?'+urllib.parse.urlencode({'dataset':'google/fleurs','config':language,'split':'test'}))]:
        try:
            with urllib.request.urlopen(url,timeout=30) as r: data=json.load(r)
            if kind=='rows':
                out={'keys':list(data),'row_keys':list(data.get('rows',[{'row':{}}])[0]['row'])}
            else: out=data
            print(json.dumps({'language':language,'kind':kind,'result':out}),flush=True)
        except Exception as e: print(json.dumps({'language':language,'kind':kind,'error':str(e).split('?')[0][:300]}),flush=True)
try:
    info = api.dataset_info('google/fleurs', revision='refs/convert/parquet')
    print(json.dumps({'converted_commit':info.sha}),flush=True)
    for language in ['cmn_hans_cn','ar_eg','de_de']:
        files=[x.path for x in api.list_repo_tree('google/fleurs',repo_type='dataset',revision=info.sha,path_in_repo=language,recursive=True) if x.path.endswith('.parquet') and '/test/' in x.path]
        print(json.dumps({'language':language,'files':files}),flush=True)
        if files:
            with fs.open(f'datasets/google/fleurs@{info.sha}/{files[0]}','rb',cache_type='none') as f:
                p=pq.ParquetFile(f)
                print(json.dumps({'language':language,'schema':str(p.schema),'groups':p.num_row_groups,'first_group_rows':p.metadata.row_group(0).num_rows}),flush=True)
except Exception as e: print(json.dumps({'converted_error':str(e).split('?')[0][:500]}),flush=True)
