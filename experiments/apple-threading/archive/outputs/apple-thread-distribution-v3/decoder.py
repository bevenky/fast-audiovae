"""Short complete-decoder comparison against the retained v2 four-worker path."""
import common as c
import json,statistics
def main():
    micro=json.loads((c.HERE/'micro-results.json').read_text());assert micro['status']=='passed'
    grain=micro['selected_grain'];receipt=c.artifact_receipt();full=c.e.graph();models={}
    for kind in ('frozen','baseline','paired'):
        models[kind]=c.e.StreamingDecoder(c.session(c.transform(full,kind,grain),old=kind=='frozen'),c.e.SPEC)
    ids=['bn_in_00151_1818','en_us_00103_1779','es_419_00060_1994']
    with c.np.load(c.e.CFG['latents'],allow_pickle=False) as z:
        inputs={uid:c.np.ascontiguousarray(z[uid+'__z'][...,:24]) for uid in ids}
    with c.np.load(c.e.CFG['extra_latents'],allow_pickle=False) as z:
        edges=[c.np.ascontiguousarray(z[k+'__z'][...,:3]) for k in ('panel_000','digital_zero_1s','quiet_speech_6s')]
    edges.append(c.np.zeros((1,64,3),dtype='f'))
    for z in [*inputs.values(),*edges]:
        for packet in (1,2,4):
            ref,_,_=c.timed(lambda:c.e.packet_run(models['frozen'],z,packet,True))
            for name in ('baseline','paired'):
                actual,_,_=c.timed(lambda:c.e.packet_run(models[name],z,packet,True));c.compare_run(ref,actual)
    rows=[]
    for packet in (1,2):
        for uid,z in inputs.items():
            for _ in range(2):
                for name in ('baseline','paired'):c.timed(lambda:c.e.packet_run(models[name],z,packet))
            for rep in range(4):
                order=['baseline','paired'] if rep%2==0 else ['paired','baseline'];row={'packet_ms':packet*40,'uid':uid,'repeat':rep,'order':order}
                for name in order:
                    _,wall,cpu=c.timed(lambda:c.e.packet_run(models[name],z,packet));row[name]={'wall':wall,'cpu':cpu,'rtf':wall/.96}
                row['reduction_percent']=100*(1-row['paired']['wall']/row['baseline']['wall']);rows.append(row)
    summary=[]
    for ms in (40,80):
        rs=[v for v in rows if v['packet_ms']==ms];vals=[v['reduction_percent'] for v in rs]
        summary.append({'packet_ms':ms,'median_paired_reduction_percent':statistics.median(vals),'wins':sum(x>0 for x in vals),'pairs':len(vals),
            'baseline_rtf':statistics.median(v['baseline']['rtf'] for v in rs),'paired_rtf':statistics.median(v['paired']['rtf'] for v in rs),
            'baseline_cpu_per_wall':sum(v['baseline']['cpu'] for v in rs)/sum(v['baseline']['wall'] for v in rs),
            'paired_cpu_per_wall':sum(v['paired']['cpu'] for v in rs)/sum(v['paired']['wall'] for v in rs)})
    assert receipt==c.artifact_receipt()
    c.RESULT.update(status='passed',grain=grain,summary=summary,rows=rows,artifacts=receipt,
        scope='same-build first-pair16+lateDW baseline versus added second-pair flat scheduling; three960ms prefixes;fourbalancedpairs each;4CPUthreads; no diagnostic state copies during timing')
    (c.HERE/'decoder-results.json').write_text(json.dumps(c.RESULT,indent=2)+'\n');print(json.dumps({k:v for k,v in c.RESULT.items() if k not in ('rows','artifacts')},indent=2))
if __name__=='__main__':main()
