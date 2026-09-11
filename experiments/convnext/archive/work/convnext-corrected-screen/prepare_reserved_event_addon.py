import hashlib,json,math
from pathlib import Path
import numpy as np,soundfile as sf
B=Path('/workspace/fast-audiovae-convnext-20260909-r9/remediation')
p=B/'corrected-reserved-event-candidates.json';doc=json.loads(p.read_text());out=[];seen=set()
for x in doc['candidates']:
 r=x['row'];sid=r['source_id']
 if sid in seen:continue
 seen.add(sid);assert not x['overlap_training_identity']
 raw=Path(r['audio_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['audio_sha256']
 a,sr=sf.read(r['audio_path'],dtype='float32',always_2d=True)
 assert sr==16000 and a.shape==(round(r['duration_seconds']*16000),1) and np.isfinite(a).all()
 out.append({'row':r,'source_labels':x['labels'],'metadata':{'dataset':r['dataset'],'language':r['language'],'condition':'Yell','event':'Yell','speech':False,'semantic_event_verified':False},
  'crops':[{'start_frame':0,'scored_frames':math.ceil(len(a)/640),'context_frames':0,'valid_scored_samples':len(a)*3}],
  'qualification':{'input_rms':float(np.sqrt(np.mean(a.astype('float64')**2))),'input_peak':float(np.max(np.abs(a))),
   'evidence':'Two positive reviewed source-level Yell ratings; no temporal annotation; full utterance retained',
   'source_metadata':x['metadata'],'parent_or_training_identity_overlap':False}})
res={'format_version':1,'scope':'Additional development source, separate from unchanged primary panel',
 'source_inventory_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'entries':out,
 'missing_verified_source_categories':['Crying_and_sobbing','Giggle','Shout'],
 'one_yell_source_is_not_population_quality_estimate':True}
path=B/'corrected-screen/addon-validation.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(res,indent=2)+'\n');print(path)
