"""Run bounded diagnostics sequentially on one H100; keep retained files intact."""
import fcntl,gc,json,os,time
from pathlib import Path
import torch
from diagnostic_common import load_context,atomic_json,status,BASE,file_sha

def main():
    out=BASE/'diagnostics-v1'
    with (BASE.parent/'training-runs/.decoder-recipe-v2-expressive.runner.lock').open('rb') as original, (out/'runner.lock').open('a') as own:
        fcntl.flock(original.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(own.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        ctx=load_context()
        sources={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}
        receipt={'sources':sources,'checkpoints':ctx.receipt['checkpoints'],'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        if (out/'run-identity.json').exists():raise RuntimeError('Refusing automatic diagnostic replay')
        atomic_json(out/'run-identity.json',receipt)
        import diagnose_objectives_updates,diagnose_features,diagnose_history_sensitivity
        for name,module in [('objectives_updates',diagnose_objectives_updates),('features',diagnose_features),('history_sensitivity',diagnose_history_sensitivity)]:
            status('starting',diagnostic=name);start=time.monotonic()
            module.run(ctx)
            atomic_json(out/(name+'-complete.json'),{'seconds':time.monotonic()-start,**ctx.verify_files()})
            gc.collect();torch.cuda.empty_cache();status('completed',diagnostic=name)
        if {p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}!=sources:raise RuntimeError('Diagnostic source changed during run')
        atomic_json(out/'complete.json',{'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    'original_training_run':'paused',**ctx.verify_files(),'retained_model_updates':0})
        status('all_complete')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(BASE/'diagnostics-v1/failure.json',{'error':repr(exc),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        raise
