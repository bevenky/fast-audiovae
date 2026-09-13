"""Three instrumented calls to observe the selected four-worker distribution."""
import common as c
import micro,json,ctypes
def main():
    selected=json.loads((c.HERE/'micro-results.json').read_text());assert selected['status']=='passed';grain=selected['selected_grain']
    full=c.e.graph();lib=ctypes.CDLL(str(c.BUILD/'libdist3_paired_ort.dylib'))
    get=lib.av_paired_second_trace_get;get.argtypes=[ctypes.c_size_t,ctypes.c_size_t];get.restype=ctypes.c_uint64
    x=c.np.random.default_rng(10).normal(0,.3,(1,1024,16)).astype('f');reports=[]
    s=c.session(micro.model(full,'paired',grain,trace=1))
    for rep in range(3):
        _,_,_=c.timed(lambda:s.run(None,{'x':x}))
        a=[get(f,0) for f in range(14)];assert a[0]==1 and a[10]==0 and a[3]==16
        scale=a[12]/a[13]/1000
        tasks=[{'worker':get(32,i),'start':get(33,i),'end':get(34,i),'branch':get(35,i),'first':get(36,i),'count':get(37,i),'status':get(38,i)} for i in range(a[1])]
        assert all(t['status']==0 and t['end']>=t['start'] for t in tasks)
        for branch in (0,1):assert sorted(p for t in tasks if t['branch']==branch for p in range(t['first'],t['first']+t['count']))==list(range(3072))
        workers={}
        for t in tasks:workers[t['worker']]=workers.get(t['worker'],0)+(t['end']-t['start'])*scale
        events=sorted([(t['start'],1) for t in tasks]+[(t['end'],-1) for t in tasks]);active=0;previous=a[6];area=0
        for when,delta in events:area+=(when-previous)*active;active+=delta;previous=when
        dispatch=(a[7]-a[6])*scale
        reports.append({'grain':grain,'repeat':rep,'tasks':a[1],'workers':len(workers),'peak_callbacks':a[9],
            'prepare_us':(a[6]-a[5])*scale,'dispatch_us':dispatch,'finish_us':(a[8]-a[7])*scale,
            'first_callback_delay_us':(min(t['start'] for t in tasks)-a[6])*scale,
            'after_last_callback_us':(a[7]-max(t['end'] for t in tasks))*scale,
            'average_active_callbacks':area*scale/dispatch,'per_worker_callback_us':sorted(workers.values()),'raw_summary':a,'raw_tasks':tasks})
    c.RESULT.update(status='passed',traces=reports,scope='three tracedM16 calls; callback overlap not hardware residency; trace disabled during performance tests')
    (c.HERE/'trace-results.json').write_text(json.dumps(c.RESULT,indent=2)+'\n')
    print(json.dumps({**c.RESULT,'traces':[{k:v for k,v in t.items() if not k.startswith('raw_')} for t in reports]},indent=2))
if __name__=='__main__':main()
