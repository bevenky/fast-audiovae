import json
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
fs=HfFileSystem()
for lang in ['cmn_hans_cn','ar_eg','de_de']:
    with fs.open(f'datasets/google/fleurs@168de341b3db6859a9bac1c50a2ef5e3b47647e0/{lang}/test/0000.parquet','rb',cache_type='none') as f:
        p=pq.ParquetFile(f)
        g=p.metadata.row_group(0)
        rows=p.read_row_group(0,columns=['id','num_samples','path','audio.path','gender','language']).to_pylist()
        print(json.dumps({'language':lang,'audio_compressed_bytes':sum(g.column(j).total_compressed_size for j in range(g.num_columns) if g.column(j).path_in_schema=='audio.bytes'),'rows':rows[:5]}),flush=True)
