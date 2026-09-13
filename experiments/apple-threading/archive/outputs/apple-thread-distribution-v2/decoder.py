"""Short normal streaming API timing, separate from full-state parity checks."""
import experiment as e
import json,statistics
def main():
    micro=json.loads((e.HERE/'micro-results.json').read_text());assert micro['status']=='passed'
    # Operator screening selected finer sixteen tasks across the two branches.
    # Confirm only this candidate against the previous combined four-thread path.
    grain=16;full=e.graph();models={}
    for kind in ('frozen','separate','paired'):
        models[kind]=e.StreamingDecoder(e.session(e.transform(full,kind,grain),old=kind=='frozen'),e.SPEC)
    ids=['bn_in_00151_1818','en_us_00103_1779','es_419_00060_1994']
    with e.np.load(e.CFG['latents'],allow_pickle=False) as z:
        inputs={uid:e.np.ascontiguousarray(z[uid+'__z'][...,:24]) for uid in ids}
    with e.np.load(e.CFG['extra_latents'],allow_pickle=False) as z:
        edges=[e.np.ascontiguousarray(z[k+'__z'][...,:3]) for k in ('panel_000','digital_zero_1s','quiet_speech_6s')]
    edges.append(e.np.zeros((1,64,3),dtype='f'))
    for z in [*inputs.values(),*edges]:
        for packet in (1,2,4):
            ref,_,_=e.timed(lambda:e.packet_run(models['frozen'],z,packet,True))
            for name in ('separate','paired'):
                actual,_,_=e.timed(lambda:e.packet_run(models[name],z,packet,True));e.compare_run(ref,actual)
    # Timing omits all per-packet diagnostic state copies and comparisons.
    rows=[]
    for packet in (1,2):
        for uid,z in inputs.items():
            for _ in range(2):
                for name in ('separate','paired'):e.timed(lambda:e.packet_run(models[name],z,packet))
            for rep in range(4):
                order=['separate','paired'] if rep%2==0 else ['paired','separate'];row={'packet_ms':packet*40,'uid':uid,'repeat':rep}
                for name in order:
                    _,wall,cpu=e.timed(lambda:e.packet_run(models[name],z,packet));row[name]={'wall':wall,'cpu':cpu,'rtf':wall/.96}
                row['reduction_percent']=100*(1-row['paired']['wall']/row['separate']['wall']);rows.append(row)
    summary=[]
    for ms in (40,80):
        r=[v for v in rows if v['packet_ms']==ms];vals=[v['reduction_percent'] for v in r]
        summary.append({'packet_ms':ms,'median_paired_reduction_percent':statistics.median(vals),'wins':sum(x>0 for x in vals),'pairs':len(vals),
            'separate_rtf':statistics.median(v['separate']['rtf'] for v in r),'paired_rtf':statistics.median(v['paired']['rtf'] for v in r),
            'separate_cpu_per_wall':sum(v['separate']['cpu'] for v in r)/sum(v['separate']['wall'] for v in r),
            'paired_cpu_per_wall':sum(v['paired']['cpu'] for v in r)/sum(v['paired']['wall'] for v in r)})
    e.RESULT.update(status='passed',grain=grain,summary=summary,rows=rows,scope='same-build previous combined baseline versus flat paired distribution; three960ms prefixes;4paired repetitions;4CPUthreads; normal API timing without diagnostic state snapshots; parity separate')
    (e.HERE/'decoder-results.json').write_text(json.dumps(e.RESULT,indent=2)+'\n');print(json.dumps({k:v for k,v in e.RESULT.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
