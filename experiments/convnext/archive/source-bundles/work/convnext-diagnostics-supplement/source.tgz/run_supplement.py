"""Close three measured diagnostic limitations without changing retained models."""
import fcntl,gc,time
from pathlib import Path
import torch
from diagnostic_common import BASE,load_context,atomic_json,status,file_sha

def main():
    out=BASE/'diagnostics-v1'
    with (BASE.parent/'training-runs/.decoder-recipe-v2-expressive.runner.lock').open('rb') as original, (out/'runner.lock').open('a') as own:
        fcntl.flock(original.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(own.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (out/'complete.json').exists():raise RuntimeError('Main diagnostic suite must finish first')
        if (out/'supplement-identity.json').exists():raise RuntimeError('Refusing automatic supplement replay')
        ctx=load_context()
        sources={str(p):file_sha(p) for root in (Path(__file__).parent,BASE/'diagnostic-code') for p in root.glob('*.py')}
        atomic_json(out/'supplement-identity.json',{'sources':sources,'reason':'Natural history coverage; bounded ridge grid; finite native update localization',
                    'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checkpoints':ctx.receipt['checkpoints']})
        import natural_history,diagnose_step_path,diagnose_ridge_extension
        for name,module in [('natural_history',natural_history),('native_step_path',diagnose_step_path),('ridge_extension',diagnose_ridge_extension)]:
            status('starting_supplement',diagnostic=name);start=time.monotonic()
            module.run(ctx)
            atomic_json(out/(name+'-complete.json'),{'seconds':time.monotonic()-start,**ctx.verify_files()})
            gc.collect();torch.cuda.empty_cache();status('completed_supplement',diagnostic=name)
        if {p:file_sha(Path(p)) for p in sources}!=sources:raise RuntimeError('Diagnostic source changed')
        atomic_json(out/'supplement-complete.json',{'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    'original_training_run':'paused',**ctx.verify_files(),'retained_model_updates':0})
        status('supplement_all_complete')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(BASE/'diagnostics-v1/supplement-failure.json',{'error':repr(exc)})
        raise
