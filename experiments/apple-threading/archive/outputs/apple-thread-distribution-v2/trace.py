"""Observe worker overlap, separately from uninstrumented speed measurements."""
import experiment as e
import micro,json,ctypes
def main():
    full=e.graph();lib=ctypes.CDLL(str(e.BUILD/'libdist_paired_ort.dylib'))
    get=lib.av_paired_first_trace_get;get.argtypes=[ctypes.c_size_t,ctypes.c_size_t];get.restype=ctypes.c_uint64
    x=e.np.random.default_rng(10).normal(0,.3,(1,2048,2)).astype('f');reports=[]
    for grain in (16,32,64):
        s=e.session(micro.model(full,'paired',grain,trace=1))
        for rep in range(3):
            _,_,_=e.timed(lambda:s.run(None,{'x':x}))
            a=[get(f,0) for f in range(14)];assert a[0]==1 and a[10]==0 and a[3]==2
            scale=a[12]/a[13]/1000
            tasks=[{'worker':get(32,i),'start':get(33,i),'end':get(34,i),'branch':get(35,i),'first':get(36,i),'count':get(37,i),'status':get(38,i)} for i in range(a[1])]
            assert all(t['status']==0 and t['end']>=t['start'] for t in tasks)
            for branch in (0,1):assert sorted(p for t in tasks if t['branch']==branch for p in range(t['first'],t['first']+t['count']))==list(range(128))
            by_worker={}
            for t in tasks:by_worker[t['worker']]=by_worker.get(t['worker'],0)+(t['end']-t['start'])*scale
            first=min(t['start'] for t in tasks);last=max(t['end'] for t in tasks)
            events=sorted([(t['start'],1) for t in tasks]+[(t['end'],-1) for t in tasks]);active=0;previous=a[6];area=0;nonzero=0
            for when,delta in events:
                area+=(when-previous)*active;nonzero+=(when-previous)*(active>0);active+=delta;previous=when
            dispatch=(a[7]-a[6])*scale
            reports.append({'grain':grain,'repeat':rep,'tasks':a[1],'workers':len(by_worker),'peak_callbacks':a[9],
                'prepare_us':(a[6]-a[5])*scale,'dispatch_us':dispatch,'finish_us':(a[8]-a[7])*scale,
                'first_callback_delay_us':(first-a[6])*scale,'after_last_callback_us':(a[7]-last)*scale,
                'average_active_callbacks':area*scale/dispatch,'per_worker_callback_us':sorted(by_worker.values()),
                'branch_callback_us':[sum((t['end']-t['start'])*scale for t in tasks if t['branch']==b) for b in (0,1)]})
    e.RESULT.update(status='passed',traces=reports,scope='nine instrumented M2 calls; callback overlap is not proof of four cores sustained execution; clocks excluded from speed trials')
    (e.HERE/'trace-results.json').write_text(json.dumps(e.RESULT,indent=2)+'\n');print(json.dumps(e.RESULT,indent=2))
if __name__=='__main__':main()
