"""Verify completed AMD exports against previously scored Intel FLOAT WAV bytes."""
import argparse,hashlib,json,struct
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wav_payload(path):
    data=path.read_bytes();assert data[:4]==b'RIFF' and data[8:12]==b'WAVE';i=12;parts={}
    while i+8<=len(data):
        tag=data[i:i+4];size=struct.unpack_from('<I',data,i+4)[0];assert i+8+size<=len(data);assert tag not in parts;parts[tag]=data[i+8:i+8+size];i+=8+size+(size&1)
    assert i==len(data);fmt=struct.unpack_from('<HHIIHH',parts[b'fmt ']);assert fmt[0]==3 and fmt[1]==1 and fmt[2]==48000 and fmt[5]==32 and len(parts[b'data'])%4==0
    return hashlib.sha256(parts[b'data']).hexdigest(),len(parts[b'data'])//4

def main():
    p=argparse.ArgumentParser();p.add_argument('--amd-report',type=Path,required=True);p.add_argument('--intel-transfer',type=Path,required=True);p.add_argument('--intel-campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    report=json.loads(a.amd_report.read_text());transfer=json.loads(a.intel_transfer.read_text());assert report['status']=='complete' and report['mode']=='full_validation_only' and report['platform']=='amd';assert len(report['checks'])==234 and len(report['exports'])==180 and not report['measurements'] and not report['failures'] and all(x['passed'] for x in report['checks'])
    uids=report['protocol']['validation_uids'];assert len(uids)==len(set(uids))==60
    outputs={(r['model'],r['uid']):r for r in report['exports']};checks={(r['model'],r['uid']):r for r in report['checks'] if r['test']=='full_waveform'};old={(r['model'],r['uid']):r for r in transfer['exports']};assert len(outputs)==180 and len(checks)==180 and len(old)==len(transfer['exports'])
    mapping={'audio_stock':'stock_fp32','fast_fp32':'fast_fp32','int8_large':'int8_large'};rows=[];summary={}
    for name,previous in mapping.items():
        matches=0;filematches=0
        for uid in uids:
            new=outputs[name,uid];check=checks[name,uid];record=old[previous,uid];path=a.intel_campaign/record['path'];assert sha(path)==record['sha256'],'Previously scored waveform file changed'
            payload,samples=wav_payload(path);assert new['rate']==record['rate']==48000 and new['subtype']==record['subtype']=='FLOAT' and new['samples']==record['samples']==samples
            same=payload==check['waveform_sha256'];wholefile=new['sha256']==record['sha256'];assert not wholefile or same
            matches+=same;filematches+=wholefile;rows.append({'model':name,'uid':uid,'samples':samples,'rate':48000,'AMD_waveform_sha256':check['waveform_sha256'],'AMD_file_sha256':new['sha256'],'Intel_waveform_sha256':payload,'Intel_file_sha256':record['sha256'],'waveform_bitwise_equal':same,'file_byte_equal':wholefile,'source_latent_sha256':check['latent_sha256']})
        summary[name]={'compared':60,'identical_waveforms':matches,'identical_files':filematches,'quality_score_reuse_supported':matches==60}
    result={'status':'complete','scope':'Exact sample/file identity evidence only, no new predictor inference or perceptual equivalence claim between INT8 and FP32','AMD_report_sha256':sha(a.amd_report),'AMD_config_sha256':report['config_sha256'],'Intel_transfer_sha256':sha(a.intel_transfer),'Intel_source_config_sha256':transfer['config_sha256'],'records':rows,'summary':summary,'script_sha256':sha(Path(__file__)),'verification':'AMD runner freshly exported and reloaded180 FLOAT WAVs and required uint32 identity with each validated decoder waveform. Intel files were independently hashed against the earlier transfer manifest, then their IEEE-float DATA payload hashes compared to the AMD validated waveform hashes.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
