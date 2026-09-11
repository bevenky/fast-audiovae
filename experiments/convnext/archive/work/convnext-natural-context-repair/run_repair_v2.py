"""Complete natural coverage missing from the earlier diagnostic selectors."""
import fcntl,json,time
from pathlib import Path
from diagnostic_common import BASE,load_context,atomic_json,status,file_sha
import repair_natural_history

def main():
    out=BASE/'diagnostics-v1'
    with (BASE.parent/'training-runs/.decoder-recipe-v2-expressive.runner.lock').open('rb') as original,(out/'runner.lock').open('a') as own:
        fcntl.flock(original.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(own.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'natural-repair-identity.json').exists():raise RuntimeError('Refusing replay')
        ctx=load_context()
        inventory=repair_natural_history.run(ctx,inventory_only=True)
        atomic_json(out/'natural-repair-inventory.json',inventory)
        raw_groups={c['category'] for c in inventory['inventory']['cases'] if 'amplitude_source' in c}
        if inventory['selection']['missing_categories'] or inventory['inventory']['amplitude_available_cases']<7 or len(raw_groups)!=4:
            raise RuntimeError('Natural probe coverage or authenticated raw source coverage incomplete')
        sources={str(p):file_sha(p) for root in (Path(__file__).parent,BASE/'diagnostic-code',BASE/'diagnostic-supplement-code') for p in root.glob('*.py')}
        atomic_json(out/'natural-repair-identity.json',{'sources':sources,'checkpoints':ctx.receipt['checkpoints'],
                    'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        repair_natural_history.run(ctx)
        if {p:file_sha(Path(p)) for p in sources}!=sources:raise RuntimeError('Diagnostic source changed')
        atomic_json(out/'natural-repair-complete.json',{'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    **ctx.verify_files(),'retained_model_updates':0,'original_training_run':'paused'})
        status('natural_repair_all_complete')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(BASE/'diagnostics-v1/natural-repair-failure.json',{'error':repr(exc)})
        raise
