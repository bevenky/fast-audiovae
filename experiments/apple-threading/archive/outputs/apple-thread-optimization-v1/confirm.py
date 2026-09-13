"""One bounded confirmation of the combined80ms candidate against same-build serial."""
from pathlib import Path
import json,random,statistics,time
import decoder as d

def main():
    source=json.loads((d.HERE/'decoder-results.json').read_text());assert source['status']=='passed'
    model=d.onnx.load(d.BUNDLE/d.SPEC['model'])
    models={};metadata={}
    for k in ('serial','combined'):
        models[k],metadata[k]=d.make(model,k,4,source['matrix_grain'],source['channel_blocks'])
    report={'status':'running','metadata':metadata,'scope':'80ms only, same-build control, two warmups per input/arm, four balanced paired repetitions on each of three960ms prefixes; no spins;12s call cap','rows':[]}
    inputs={}
    with d.np.load(d.CFG['latents'],allow_pickle=False) as a:
        for uid in ('bn_in_00151_1818','en_us_00103_1779','es_419_00060_1994'):
            inputs[uid]=d.np.ascontiguousarray(a[uid+'__z'][...,:24])
    for uid,z in inputs.items():
        reference=d.one(models['serial'],z,2)
        for _ in range(2):
            for k in models:d.one(models[k],z,2,reference)
        for repeat in range(4):
            order=['serial','combined'] if repeat%2==0 else ['combined','serial']
            row={'uid':uid,'repeat':repeat,'order':order}
            for k in order:
                _,_,seconds,cpu=d.one(models[k],z,2,reference)
                row[k]={'seconds':seconds,'rtf':seconds/.96,'cpu_seconds':cpu}
                assert d.RESULT['call_seconds']<12
            row['reduction_percent']=100*(1-row['combined']['seconds']/row['serial']['seconds'])
            report['rows'].append(row)
    reductions=[r['reduction_percent'] for r in report['rows']]
    report.update(status='passed',median_reduction_percent=statistics.median(reductions),
        wins=sum(v>0 for v in reductions),max_abs=d.RESULT['max_abs'],checks=d.RESULT['checks'],call_seconds=d.RESULT['call_seconds'])
    for k in models:
        report[k+'_rtf']=statistics.mean(statistics.median(r[k]['rtf'] for r in report['rows'] if r['uid']==u) for u in inputs)
    (d.HERE/'confirmation-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','metadata')},indent=2))
if __name__=='__main__':main()
