from pathlib import Path
import os,time,json,hashlib,shutil
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES']='0'
base=Path('/workspace/fast-audiovae-convnext-20260908-r1');out=base/'indic-io-probe';out.mkdir(exist_ok=True)
if (out/'native').exists():raise SystemExit('Probe-owned native staging already exists; inspect before retry')
from huggingface_hub import HfApi,HfFileSystem,hf_hub_download
from huggingface_hub.hf_file_system import HfFileSystemFile
import pyarrow.parquet as pq
from audiovae_student.acquire_indic import REPO,REVISION,_COLUMNS
filename='bengali/train-00020-of-00096.parquet'
info=HfApi().get_paths_info(REPO,[filename],repo_type='dataset',revision=REVISION)[0]
if not 0<info.size<600*2**20 or shutil.disk_usage(out).free-info.size<8*2**30:raise SystemExit('Probe byte/reserve guard')
report={'repo':REPO,'revision':REVISION,'filename':filename,'shard_bytes':info.size,'created_at':time.time(),'pre_buffer':True,'range_events':[],'native_chunk_cache_bytes':0}
original=HfFileSystemFile._fetch_range
phase='open';total_requested=0

def record(self,start,end):
 global total_requested
 if total_requested+(end-start)>150*2**20:raise ValueError('Probe range byte limit exceeded')
 total_requested+=end-start
 t=time.perf_counter();payload=original(self,start,end)
 report['range_events'].append({'phase':phase,'start':start,'end':end,'bytes':len(payload),'seconds':time.perf_counter()-t})
 return payload
HfFileSystemFile._fetch_range=record
try:
 fs=HfFileSystem();t=time.perf_counter()
 with fs.open(f'datasets/{REPO}@{REVISION}/{filename}','rb',block_size=1024**2,cache_type='none') as stream:
  parquet=pq.ParquetFile(stream,pre_buffer=True)
  report['range_open_seconds']=time.perf_counter()-t
  groups=[]
  for n in range(parquet.num_row_groups):
   rg=parquet.metadata.row_group(n)
   size=sum(rg.column(i).total_compressed_size for i in range(rg.num_columns) if rg.column(i).path_in_schema=='audio_filepath.bytes')
   groups.append({'group':n,'rows':rg.num_rows,'audio_column_bytes':size})
  report['groups']=groups
  group=next((g['group'] for g in groups if g['audio_column_bytes']<110*2**20),None)
  if group is None:raise ValueError('No bounded rowgroup')
  report['group']=group
  phase='metadata';t=time.perf_counter();metadata=parquet.read_row_group(group,columns=_COLUMNS).to_pylist();report['range_metadata_seconds']=time.perf_counter()-t
  print(json.dumps({'phase':'metadata_complete','seconds':report['range_metadata_seconds'],'requests':len(report['range_events'])}),flush=True)
  phase='audio';t=time.perf_counter();blobs=parquet.read_row_group(group,columns=['audio_filepath.bytes'])['audio_filepath'].to_pylist();report['range_audio_seconds']=time.perf_counter()-t
  payload_hashes=[hashlib.sha256(x['bytes']).hexdigest() for x in blobs];del blobs
  report['range_total_bytes']=sum(x['bytes'] for x in report['range_events']);report['range_request_count']=len(report['range_events'])
  print(json.dumps({'phase':'range_complete','audio_seconds':report['range_audio_seconds'],'bytes':report['range_total_bytes'],'requests':report['range_request_count']}),flush=True)
 HfFileSystemFile._fetch_range=original
 t=time.perf_counter();native=Path(hf_hub_download(REPO,filename,repo_type='dataset',revision=REVISION,local_dir=out/'native'));report['native_download_seconds']=time.perf_counter()-t
 print(json.dumps({'phase':'native_download_complete','seconds':report['native_download_seconds'],'bytes':native.stat().st_size}),flush=True)
 if native.stat().st_size!=info.size:raise ValueError('Native shard size mismatch')
 with native.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 expected=info.lfs.sha256 if hasattr(info.lfs,'sha256') else info.lfs['sha256']
 if actual!=expected:raise ValueError('Native shard SHA mismatch')
 report['shard_sha256']=actual;report['upstream_sha256_verified']=True
 t=time.perf_counter();local=pq.ParquetFile(native);local_metadata=local.read_row_group(group,columns=_COLUMNS).to_pylist();report['native_metadata_seconds']=time.perf_counter()-t
 t=time.perf_counter();local_blobs=local.read_row_group(group,columns=['audio_filepath.bytes'])['audio_filepath'].to_pylist();report['native_audio_decode_seconds']=time.perf_counter()-t
 report['metadata_equal']=metadata==local_metadata
 report['audio_bytes_equal']=payload_hashes==[hashlib.sha256(x['bytes']).hexdigest() for x in local_blobs]
 if not report['metadata_equal'] or not report['audio_bytes_equal']:raise ValueError('Native/range rows differ')
 report['state']='complete'
except BaseException as error:
 report['state']='failed';report['error_type']=type(error).__name__
 if type(error) is ValueError:report['error']=str(error)
finally:
 HfFileSystemFile._fetch_range=original
 if (out/'native').exists():
  shutil.rmtree(out/'native');report['owned_staging_removed']=True
 report['finished_at']=time.time();(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)
if report['state']!='complete':raise SystemExit(1)
