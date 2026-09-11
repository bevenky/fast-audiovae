"""Read three authenticated source prefixes to interpret teacher startup mismatch."""
from pathlib import Path
import hashlib
import json
import numpy as np
import soundfile as sf

root=Path('/workspace/fast-audiovae-compression-20260910-v1')
manifest=json.loads((root/'pilot-selection-v1.json').read_text())
rows=[]
for prefix in ('tg_tj:', 'lb_lu:', 'so_so:'):
    row=next(r for r in manifest['splits']['development']['rows'] if r['source_id'].startswith(prefix))
    assert row['start_frame']==0 and row['context_frames']==0
    path=Path(row['manifest_row']['audio_path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['audio_sha256']
    with sf.SoundFile(path) as f:
        rate=f.samplerate
        x=f.read(round(.190*rate),dtype='float64',always_2d=True)
    rows.append({'source_id':row['source_id'],'sample_rate':rate,'frames':len(x),
                 'duration_seconds':len(x)/rate,'nonzero_samples':int(np.count_nonzero(x)),
                 'rms':float(np.sqrt(np.mean(x*x))),'peak_abs':float(np.abs(x).max()),
                 'audio_sha256':row['audio_sha256']})
out=root/'startup-source-check-v1.json'
assert not out.exists()
out.write_text(json.dumps({'neural_inference_calls':0,'rows':rows},indent=2)+'\n')
print(json.dumps(rows))
