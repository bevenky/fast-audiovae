"""Small four-worker paired-projection screen, original trained weights."""
import experiment as e
import copy,json,random,statistics
def model(full,kind,grain=32,trace=0):
    a,b=e.pair_nodes(full);a=copy.deepcopy(a);b=copy.deepcopy(b)
    a.input[0]=b.input[0]='x'
    if kind=='separate':a.attribute.append(e.helper.make_attribute('parallel_panels',32))
    nodes=[a,b] if kind in ('frozen','separate') else [e.paired(a,b,grain,serial=int(kind=='serial'),trace=trace)]
    init=[v for v in full.graph.initializer if v.name in (a.input[1],b.input[1])]
    g=e.helper.make_graph(nodes,'paired-first',[e.helper.make_tensor_value_info('x',e.TensorProto.FLOAT,['B',2048,'M'])],
        [e.helper.make_tensor_value_info(n,e.TensorProto.FLOAT,['B',8192,'M']) for n in (a.output[0],b.output[0])],init)
    m=e.helper.make_model(g,opset_imports=list(full.opset_import)+[e.helper.make_opsetid(e.DOMAIN,1)]);m.ir_version=full.ir_version
    return m
def main():
    full=e.graph();cases={'frozen':e.session(model(full,'frozen'),old=True),'separate':e.session(model(full,'separate')),
        'serial':e.session(model(full,'serial'))}
    for grain in (16,32,64):cases[f'paired{grain}']=e.session(model(full,'paired',grain))
    rng=e.np.random.default_rng(93);rows=[]
    for m in (1,2):
        x=rng.normal(0,.3,(1,2048,m)).astype('f');ref,_,_=e.timed(lambda:cases['frozen'].run(None,{'x':x}));keep=x.copy()
        for s in cases.values():
            for _ in range(2):
                out,_,_=e.timed(lambda:s.run(None,{'x':x}))
                for y,z in zip(out,ref):e.equal(y,z)
        for rep in range(6):
            order=list(cases);random.Random(m*100+rep).shuffle(order);row={'m':m,'repeat':rep}
            for name in order:
                out,wall,cpu=e.timed(lambda:cases[name].run(None,{'x':x}));row[name]={'wall':wall,'cpu':cpu}
                for y,z in zip(out,ref):e.equal(y,z)
            rows.append(row)
        e.equal(x,keep)
    # Zero, quiet, empty, odd-tail M3 and B2 preserve original fallback semantics.
    for shape,scale in [((1,2048,0),0),((0,2048,2),0),((1,2048,1),0),((1,2048,2),1e-7),((1,2048,3),.1),((2,2048,2),.2)]:
        x=(rng.normal(size=shape)*scale).astype('f');ref,_,_=e.timed(lambda:cases['frozen'].run(None,{'x':x}))
        for s in cases.values():
            out,_,_=e.timed(lambda:s.run(None,{'x':x}))
            for y,z in zip(out,ref):e.equal(y,z)
    summary=[]
    for m in (1,2):
        for name in ('serial','paired16','paired32','paired64'):
            vals=[100*(1-r[name]['wall']/r['separate']['wall']) for r in rows if r['m']==m]
            summary.append({'m':m,'candidate':name,'median_reduction_percent':statistics.median(vals),'wins':sum(v>0 for v in vals),
                'candidate_us':statistics.median(r[name]['wall'] for r in rows if r['m']==m)*1e6,
                'separate_us':statistics.median(r['separate']['wall'] for r in rows if r['m']==m)*1e6})
    e.RESULT.update(status='passed',rows=rows,summary=summary,scope='six randomized-order micro repetitions at M1/M2, six edge shapes, FP32 CPU four ORT workers')
    (e.HERE/'micro-results.json').write_text(json.dumps(e.RESULT,indent=2)+'\n')
    print(json.dumps({k:v for k,v in e.RESULT.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
