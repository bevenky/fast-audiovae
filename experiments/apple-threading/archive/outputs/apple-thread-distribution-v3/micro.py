"""Small paired stage-two screen at the actual M8 and M16 packet shapes."""
import common as c
import copy,json,random,statistics
def model(full,kind,grain=768,trace=0):
    a,b=c.pair_nodes(full);a=copy.deepcopy(a);b=copy.deepcopy(b);a.input[0]=b.input[0]='x'
    nodes=[a,b] if kind in ('frozen','baseline') else [c.node(a,b,grain,serial=int(kind=='serial'),trace=trace)]
    init=[v for v in full.graph.initializer if v.name in (a.input[1],b.input[1])]
    graph=c.helper.make_graph(nodes,'paired-second',[c.helper.make_tensor_value_info('x',c.e.TensorProto.FLOAT,['B',1024,'M'])],
        [c.helper.make_tensor_value_info(n,c.e.TensorProto.FLOAT,['B',3072,'M']) for n in (a.output[0],b.output[0])],init)
    m=c.helper.make_model(graph,opset_imports=list(full.opset_import)+[c.helper.make_opsetid(c.DOMAIN,1)]);m.ir_version=full.ir_version;return m
def main():
    receipt=c.artifact_receipt();full=c.e.graph();cases={'frozen':c.session(model(full,'frozen'),old=True),
        'baseline':c.session(model(full,'baseline')),'serial':c.session(model(full,'serial'))}
    for grain in (384,768,1536):cases[f'paired{grain}']=c.session(model(full,'paired',grain))
    rng=c.np.random.default_rng(93);rows=[]
    for m in (8,16):
        x=rng.normal(0,.3,(1,1024,m)).astype('f');ref,_,_=c.timed(lambda:cases['frozen'].run(None,{'x':x}));keep=x.copy()
        for s in cases.values():
            for _ in range(2):
                out,_,_=c.timed(lambda:s.run(None,{'x':x}))
                for y,z in zip(out,ref):c.equal(y,z)
        for rep in range(6):
            order=list(cases);random.Random(m*100+rep//2).shuffle(order)
            if rep%2:order.reverse()
            row={'m':m,'repeat':rep,'order':order}
            for name in order:
                out,wall,cpu=c.timed(lambda:cases[name].run(None,{'x':x}));row[name]={'wall':wall,'cpu':cpu}
                for y,z in zip(out,ref):c.equal(y,z)
            rows.append(row)
        c.equal(x,keep)
    for shape,scale in [((1,1024,0),0),((0,1024,16),0),((1,1024,1),0),((1,1024,8),1e-7),((1,1024,17),.1),((2,1024,16),.2)]:
        x=(rng.normal(size=shape)*scale).astype('f');ref,_,_=c.timed(lambda:cases['frozen'].run(None,{'x':x}))
        for s in cases.values():
            out,_,_=c.timed(lambda:s.run(None,{'x':x}))
            for y,z in zip(out,ref):c.equal(y,z)
    summary=[]
    for m in (8,16):
        for name in ('serial','paired384','paired768','paired1536'):
            vals=[100*(1-r[name]['wall']/r['baseline']['wall']) for r in rows if r['m']==m]
            summary.append({'m':m,'candidate':name,'median_reduction_percent':statistics.median(vals),'wins':sum(v>0 for v in vals),
                'candidate_us':statistics.median(r[name]['wall'] for r in rows if r['m']==m)*1e6,
                'baseline_us':statistics.median(r['baseline']['wall'] for r in rows if r['m']==m)*1e6})
    selected=max((s for s in summary if s['m']==16 and s['candidate'].startswith('paired')),key=lambda s:s['median_reduction_percent'])
    c.RESULT.update(status='passed',rows=rows,summary=summary,selected_grain=int(selected['candidate'][6:]),
        artifacts=receipt,scope='six balanced-order repetitions at M8/M16 plus six edge shapes, original FP32 weights, four CPU workers')
    assert receipt==c.artifact_receipt()
    (c.HERE/'micro-results.json').write_text(json.dumps(c.RESULT,indent=2)+'\n');print(json.dumps({k:v for k,v in c.RESULT.items() if k not in ('rows','artifacts')},indent=2))
if __name__=='__main__':main()
