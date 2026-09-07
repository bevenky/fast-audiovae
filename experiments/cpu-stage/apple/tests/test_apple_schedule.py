"""Offline dependency propagation for causal segment/tile scheduling.

This evaluates integer dependency sets, not audio, neural kernels or inference.
Per-time bias tokens detect accidental computation of fictitious negative-time
activations as well as missing real past inputs. The independent reference
uses full sequences and direct tap indexing; the candidate uses tiled history.
"""
import unittest

def reference(length):
    values=[1<<t for t in range(length)];units=[]
    for u,d in enumerate((1,3,9)):
        output=[]
        for t in range(length):
            bits=values[t] | (1<<((u+1)*length+t))
            for tap in range(7):
                i=t-(6-tap)*d
                if i>=0:bits|=values[i]
            output.append(bits)
        values=output;units.append(output)
    return units

def tiled(length,q,segments):
    outputs=[[None]*length for _ in range(3)]
    coverage=[0]*length
    parts=min(length,segments)
    for part in range(parts):
        first=length*part//parts;last=length*(part+1)//parts;warm=max(0,first-78)
        histories=[[0]*(6*d) for d in (1,3,9)]
        for start in range(warm,last,q):
            n=min(q,last-start);values=[1<<t for t in range(start,start+n)]
            for u,d in enumerate((1,3,9)):
                halo=6*d;row=histories[u]+values;out=[]
                for i in range(n):
                    t=start+i;bits=values[i] | (1<<((u+1)*length+t))
                    for tap in range(7):bits|=row[i+tap*d]
                    out.append(bits)
                histories[u]=row[n:n+halo]
                if len(histories[u])!=halo:raise AssertionError('Wrong history length')
                values=out
                for i in range(max(0,first-start),n):outputs[u][start+i]=values[i]
            for t in range(max(first,start),start+n):coverage[t]+=1
    return outputs,coverage

class ScheduleTests(unittest.TestCase):
    def test_global_zero_and_internal_warmup_have_identical_dependencies(self):
        lengths=(1,6,7,17,53,54,55,63,64,65,77,78,79,127,128,129,255,256,257,511,777)
        for length in lengths:
            expected=reference(length)
            for q in (64,128,256):
                for parts in (1,2,3,4,64):
                    with self.subTest(length=length,q=q,parts=parts):
                        actual,coverage=tiled(length,q,parts)
                        self.assertEqual(actual,expected)
                        self.assertEqual(coverage,[1]*length)
    def test_future_input_never_enters_dependency_set(self):
        length=257;actual,_=tiled(length,64,3)
        input_mask=(1<<length)-1
        for unit in actual:
            for t,bits in enumerate(unit):
                self.assertEqual((bits&input_mask)>>(t+1),0)

if __name__=='__main__':unittest.main()
