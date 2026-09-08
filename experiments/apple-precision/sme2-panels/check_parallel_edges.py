"""Untimed misuse/lifetime checks for the new parallel preparation protocol."""
import argparse,concurrent.futures,ctypes as ct,json
from pathlib import Path
from check import API,P,B,I,ptr,reference,exact,sha,np
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    build=json.loads(a.build.read_text());api=API(build['core_path']);rng=np.random.default_rng(42)
    w=rng.normal(size=(128,128)).astype(np.float32);w2=rng.normal(size=(129,128)).astype(np.float32)
    x=rng.normal(size=(128,257)).astype(np.float32);plan=api.create(128,128,ptr(w),8,3);plan2=api.create(129,128,ptr(w2),8,3)
    api.require(plan and plan2);checks=[]
    try:
        inp=api.allocate_input(plan,ptr(x),257,4);api.require(inp);y=np.empty((128,257),np.float32)
        try:
            assert api.run_rows(plan,inp,ptr(y),0,128)==-1;checks.append('incomplete_run_rejected')
            q=B();s=P();sums=I()
            assert api.inspect_input(inp,ct.byref(q),ct.byref(s),ct.byref(sums))==-1;checks.append('incomplete_inspect_rejected')
            jobs=api.prepare_jobs(inp);assert jobs==4
            for j in (-1,jobs):assert api.prepare_job(inp,j)==-1
            checks.append('invalid_job_rejected')
            with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
                assert list(pool.map(lambda j:api.prepare_job(inp,j),reversed(range(jobs))))==[0]*jobs
            assert api.check_packed_input(inp)==0;checks.append('out_of_order_joined_pack_exact')
            assert api.prepare_job(inp,0)==-1;checks.append('duplicate_job_rejected')
            assert api.run_rows(plan,inp,ptr(y),0,128)==0;exact(y,reference(w,x));checks.append('complete_input_exact')
            y2=np.empty((129,257),np.float32);assert api.run_rows(plan2,inp,ptr(y2),0,129)==0
            exact(y2,reference(w2,x));checks.append('shared_input_different_plan_exact')
        finally:api.destroy_input(inp)
        bad=x.copy();bad[-1,-1]=np.nan;inp=api.allocate_input(plan,ptr(bad),257,4);api.require(inp)
        try:
            with concurrent.futures.ThreadPoolExecutor(4) as pool:statuses=list(pool.map(lambda j:api.prepare_job(inp,j),range(4)))
            assert statuses.count(-1)==1 and statuses.count(0)==3
            assert api.run_rows(plan,inp,ptr(y),0,128)==-1;checks.append('one_failed_job_blocks_execution')
        finally:api.destroy_input(inp)
        for jobs in (0,65):assert not api.allocate_input(plan,ptr(x),257,jobs)
        checks.append('invalid_job_count_rejected')
    finally:api.destroy_plan(plan);api.destroy_plan(plan2)
    result={'status':'passed','checks':checks,'build_sha256':sha(a.build),'script_sha256':sha(__file__),
            'GPU_used':False,'timings_collected':False};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
