"""Stage bounded official FLEURS held-out rows, retaining unknown-person limits."""
from __future__ import annotations
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
from audiovae_student.acquire import FLEURS_REVISION
from audiovae_student.data import ManifestRow,load_manifest,validate_manifest
from audiovae_student.comparison_data import known_identities,assert_comparison_disjoint
from acquire_heldout_languages import ROOT, digest, check_budget, write

PARQUET_REVISION='168de341b3db6859a9bac1c50a2ef5e3b47647e0'
LANGS={'cmn_hans_cn':'cmn','ar_eg':'ar','de_de':'de'}
MAX_AUDIO_COLUMN=280*1024**2


def main():
    if (ROOT/'language-ready.json').exists(): raise RuntimeError('Do not overwrite an already sealed language appendix')
    report=json.loads((ROOT/'indic-report.json').read_text())
    report['validation'].pop('training_or_teacher_unchanged',None)
    report['validation']['no_training_or_teacher_operations_requested']=True
    write(ROOT/'indic-report.json',report)
    excluded=[]
    required={'train-manifest.jsonl','reserved.jsonl','excluded-sources.jsonl'}
    for item in report['exclusion_files']:
        path=Path(item['path'])
        if digest(path)!=item['sha256']: raise RuntimeError('Exclusion snapshot changed before FLEURS staging')
        rows=load_manifest(path)
        excluded.extend(rows if '/data/recipe-v2/' in str(path) and path.name in required else [r for r in rows if r.split!='train'])
    indic=load_manifest(ROOT/'indic-manifest.jsonl')
    forbidden=set().union(*(known_identities(r) for r in excluded+indic))
    picked=[]
    fs=HfFileSystem()
    with fs.open(f'datasets/google/fleurs@{FLEURS_REVISION}/README.md','rb',cache_type='none') as f: card=f.read(2*1024**2)
    if b'cc-by-4.0' not in card.lower(): raise RuntimeError('Official pinned card does not establish reviewed license')
    write(ROOT/'fleurs-pinned-README.md',card)
    audit={'source_revision':FLEURS_REVISION,'parquet_revision':PARQUET_REVISION,'languages':{},
           'identity_limit':'FLEURS does not publish actual speaker/session IDs. Recording, exact hash and all available known identities are disjoint; unknown speaker/session independence is not claimed.',
           'identity_policy':'Existing comparison_data.known_identities omits only the explicit language-wide FLEURS unknown-session placeholder; source metadata remains unchanged.',
           'read_group_limit_bytes':MAX_AUDIO_COLUMN}
    for config,language in LANGS.items():
        entry={'language':language,'selected':0,'attempts':[]}
        audit['languages'][config]=entry
        rejected=Counter()
        # Validation avoids enormous one-group test files for Chinese/German.
        for partition in ['validation','test']:
            if entry['selected']>=4: break
            parquet_path=f'{config}/{partition}/0000.parquet'
            try:
                with fs.open(f'datasets/google/fleurs@{PARQUET_REVISION}/{parquet_path}','rb',block_size=1024**2,cache_type='none') as stream:
                    parquet=pq.ParquetFile(stream)
                    for group_index in range(min(parquet.num_row_groups,2)):
                        if entry['selected']>=4: break
                        g=parquet.metadata.row_group(group_index)
                        nbytes=sum(g.column(j).total_compressed_size for j in range(g.num_columns) if g.column(j).path_in_schema=='audio.bytes')
                        entry['attempts'].append({'partition':partition,'row_group':group_index,'audio_compressed_bytes':nbytes})
                        if nbytes>MAX_AUDIO_COLUMN:
                            rejected['audio_group_over_bounded_read_limit']+=1;continue
                        metadata=parquet.read_row_group(group_index,columns=['id','num_samples','path','audio.path','gender','language']).to_pylist()
                        candidates=[]
                        for index,item in enumerate(metadata):
                            filename=item['audio']['path']
                            if Path(filename).name!=filename or not filename.endswith('.wav'): raise RuntimeError('Unexpected original FLEURS filename')
                            parent=f'fleurs:{config}:audio:{filename}'
                            # Source and parent checks happen before any audio-column read.
                            if ('parent',parent) in forbidden:
                                rejected['existing_parent']+=1;continue
                            if not 2<=item['num_samples']/16000<=20:
                                rejected['outside_2_to_20_seconds']+=1;continue
                            if any(x[1]['id']==item['id'] for x in candidates):
                                rejected['same_transcript_within_language']+=1;continue
                            candidates.append((index,item))
                            if len(candidates)>=8:break
                        if not candidates:continue
                        check_budget()
                        blobs=parquet.read_row_group(group_index,columns=['audio.bytes'])['audio'].to_pylist()
                        for index,item in candidates:
                            if entry['selected']>=4:break
                            payload=blobs[index]['bytes'];filename=item['audio']['path']
                            with sf.SoundFile(io.BytesIO(payload)) as sound:
                                frames,sample_rate,channels=sound.frames,sound.samplerate,sound.channels
                                samples=sound.read(dtype='float32',always_2d=True)
                                subtype,format_name=sound.subtype,sound.format
                            if sample_rate!=16000 or channels!=1 or frames!=item['num_samples'] or not np.isfinite(samples).all():
                                raise RuntimeError('Original FLEURS decode/metadata mismatch; no conversion performed')
                            path=ROOT/'audio'/config/filename
                            receipt_path=ROOT/'receipts'/config/(filename+'.json')
                            row=ManifestRow(dataset='fleurs',source_revision=FLEURS_REVISION,source_id=f'{config}:{partition}:{filename}',
                                source_url=f'https://huggingface.co/datasets/google/fleurs/blob/{PARQUET_REVISION}/{parquet_path}',
                                audio_path=str(path),audio_sha256=hashlib.sha256(payload).hexdigest(),parent_recording_id=f'fleurs:{config}:audio:{filename}',
                                parent_start_seconds=0,speaker_id=None,session_id=f'fleurs:unknown-session-group:{config}',language=config,
                                sample_rate_hz=16000,original_sample_rate_hz=16000,bandwidth_hz=8000,bandwidth_class='speech_band',
                                bandwidth_evidence='Official FLEURS release delivered mono16k, with 8k Nyquist upper bound; microphone capture bandwidth is unknown',
                                native_recording=False,enhanced=False,duration_seconds=frames/16000,split='dev',source_split=partition,
                                license='CC-BY-4.0',license_url='https://creativecommons.org/licenses/by/4.0/',attribution='FLEURS: Google Research, Conneau et al.',
                                access_record=str(receipt_path),gain_policy='none: preserve original amplitude',resampler_policy='none: original delivered mono16000Hz',teacher_cache_key=None)
                            if known_identities(row)&forbidden:
                                rejected['decoded_hash_or_known_identity_collision']+=1;continue
                            validate_manifest([row])
                            assert_comparison_disjoint([row],reserved_rows=excluded+indic+picked)
                            write(path,payload)
                            if digest(path)!=row.audio_sha256: raise RuntimeError('Written source hash mismatch')
                            write(receipt_path,{'manifest':row.to_dict(),'official_source_split':partition,'source':{
                                'configuration':config,'parquet_revision':PARQUET_REVISION,'parquet_path':parquet_path,'row_group':group_index,
                                'row_number_in_group':index,'published_text_id':item['id'],'published_gender':item['gender'],
                                'original_audio_path':filename,'original_bytes':len(payload),'decoded_frames':frames,'format':format_name,'subtype':subtype},
                                'source_card_sha256':hashlib.sha256(card).hexdigest(),'speaker_session_independence':'unknown, not asserted',
                                'audio_transformations':'none: embedded delivered bytes retained unchanged'})
                            picked.append(row);forbidden.update(known_identities(row));entry['selected']+=1
                            write(ROOT/'fleurs-staged.jsonl',''.join(json.dumps(r.to_dict(),sort_keys=True)+'\n' for r in picked).encode())
                        del blobs
            except Exception as error:
                entry['attempts'].append({'partition':partition,'error_type':type(error).__name__,'error':str(error).split('?')[0][:400]})
        entry.update({'status':'ready_with_unknown_speaker_disclosure' if entry['selected']>=2 else 'coverage_gap','rejected':dict(rejected),
                      'seconds':sum(r.duration_seconds for r in picked if r.language==config)})
        write(ROOT/'fleurs-progress.json',audit)
        print(json.dumps({config:entry}),flush=True)
    if picked:
        validate_manifest(picked)
        assert_comparison_disjoint(picked,reserved_rows=excluded+indic)
        write(ROOT/'fleurs-manifest.jsonl',''.join(json.dumps(r.to_dict(),sort_keys=True)+'\n' for r in picked).encode())
    combined=indic+picked
    validate_manifest(combined)
    assert_comparison_disjoint(combined,reserved_rows=excluded)
    for row in combined:
        if digest(Path(row.audio_path))!=row.audio_sha256: raise RuntimeError('Final source hash verification failed')
    for item in report['exclusion_files']:
        if digest(Path(item['path']))!=item['sha256']:raise RuntimeError('Excluded sources changed during acquisition')
    write(ROOT/'language-manifest.jsonl',''.join(json.dumps(r.to_dict(),sort_keys=True)+'\n' for r in combined).encode())
    audit.update({'selected_recordings':len(picked),'source_schema_and_known_identity_disjointness_passed':True,
                  'entire_optimization_calibration_and_existing_heldout_excluded':True,'exclusion_files_unchanged':True,
                  'new_audio_only_on_runpod':True,'no_training_or_teacher_operations_requested':True})
    write(ROOT/'fleurs-report.json',audit)
    readiness={'status':'ready_for_separate_language_panel_review','manifest_path':str(ROOT/'language-manifest.jsonl'),
               'manifest_sha256':digest(ROOT/'language-manifest.jsonl'),'recordings':len(combined),
               'recordings_by_language':dict(Counter(r.language for r in combined)),
               'duration_seconds':sum(r.duration_seconds for r in combined),'known_speakers':len({r.speaker_id for r in combined if r.speaker_id}),
               'unknown_speaker_rows':sum(r.speaker_id is None for r in combined),'missing_requested_languages':[v for k,v in LANGS.items() if not any(r.language==k for r in picked)],
               'indic_report_path':str(ROOT/'indic-report.json'),'fleurs_report_path':str(ROOT/'fleurs-report.json'),
               'known_identity_policy':'comparison_data.known_identities; see explicit FLEURS unknown-session disclosure',
               'storage_used_bytes':check_budget()[0],'workspace_free_bytes':check_budget()[1],'completed_unix':time.time()}
    write(ROOT/'language-ready.json',readiness)
    print(json.dumps(readiness),flush=True)

if __name__=='__main__':main()
