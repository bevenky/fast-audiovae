import json
from huggingface_hub import HfApi,HfFileSystem
from urllib.request import urlopen
from urllib.error import HTTPError
import pyarrow.parquet as pq
api,fs=HfApi(),HfFileSystem()
rev='70bb2e84b976b7e960aa89f1c648e09c59f894dd'
for lang in ['cmn_hans_cn','ar_eg','de_de']:
    try:
        for base in [lang,'data/'+lang]:
            try:
                files=[x.path for x in api.list_repo_tree('google/fleurs',repo_type='dataset',revision=rev,path_in_repo=base,recursive=True) if x.path.endswith('.parquet') and 'test' in x.path]
                print(json.dumps({'lang':lang,'base':base,'files':files}),flush=True)
                for path in files[:1]:
                    with fs.open(f'datasets/google/fleurs@{rev}/{path}','rb',cache_type='none') as f:
                        p=pq.ParquetFile(f)
                        g=p.metadata.row_group(0)
                        print(json.dumps({'lang':lang,'groups':p.num_row_groups,'audio_bytes':sum(g.column(j).total_compressed_size for j in range(g.num_columns) if g.column(j).path_in_schema=='audio.bytes')}),flush=True)
            except Exception as e: print(json.dumps({'lang':lang,'base':base,'error':str(e).split('?')[0][:150]}),flush=True)
    except Exception as e: print(type(e).__name__)
for lang in ['cmn_hans_cn','de_de']:
    try:
        with urlopen(f'https://datasets-server.huggingface.co/rows?dataset=google%2Ffleurs&config={lang}&split=test&offset=1&length=4',timeout=30) as r:
            j=json.load(r)
            print(json.dumps({'lang':lang,'rows_status':'ok','keys':list(j),'row_keys':list(j['rows'][0]['row'])}),flush=True)
    except HTTPError as e: print(json.dumps({'lang':lang,'status':e.code,'body':e.read().decode()[:1000]}),flush=True)
