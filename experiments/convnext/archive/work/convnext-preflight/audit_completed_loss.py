"""Read-only CPU diagnostics of saved student checkpoints and teacher targets."""
from pathlib import Path
from collections import defaultdict
import datetime, hashlib, io, json, math, sqlite3, statistics, time
import torch
from audiovae_student.cache import UtteranceCache
from audiovae_student.model import StudentDecoder, StudentConfig
from audiovae_student.losses import WarmupReconstructionLoss
from audiovae_student.data import load_manifest

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
base = Path('/workspace/fast-audiovae-convnext-20260908-r1')
out = base / 'audit-completed-loss'
out.mkdir(exist_ok=True)
checkpoint = base / 'training-runs/warmup-speech-500h-v1/latest.pt'
final = torch.load(checkpoint, map_location='cpu', weights_only=True)
initial = torch.load(base / 'training-runs/warmup-speech-muon-v1/latest.pt', map_location='cpu', weights_only=True)
models = {}
for name, saved in [('step1000', initial), ('step10000', final)]:
    m = StudentDecoder(StudentConfig(**saved['identity']['model_config'])).eval()
    m.load_state_dict(saved['model'], strict=True)
    models[name] = m
criterion = WarmupReconstructionLoss()
db = sqlite3.connect('file:' + str(base / 'target-cache-500h/index.sqlite3') + '?mode=ro', uri=True)
entries = {json.loads(v)['row_id']: (k, json.loads(v)) for k, v in db.execute('SELECT lookup_key, info FROM entries')}
db.close()
rows = load_manifest(base / 'expanded-pilot/corpus/source-manifest.jsonl')
groups = defaultdict(list)
for row in rows:
    if row.split == 'dev' and row.source_id in entries:
        groups[(row.dataset, row.language)].append(row)
selected = []
for key, values in sorted(groups.items()):
    selected.extend(sorted(values, key=lambda r: hashlib.sha256(r.source_id.encode()).hexdigest())[:4])
report = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'device': 'cpu', 'threads': 1, 'training_updates': 0,
          'checkpoint_step': final['step'], 'selection': 'up to four deterministic hash-ranked dev clips per dataset/language group, with start and interior crops; diagnostic panel, not population quality estimate',
          'available_groups': {str(k): len(v) for k, v in groups.items()}, 'selected_rows': len(selected), 'results': []}
with checkpoint.open('rb') as f:
    report['checkpoint_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()

def numbers(values):
    return {k: float(v.detach()) for k, v in values.items()}

def waveform_stats(pred, target):
    a, b = pred.flatten().double(), target.flatten().double()
    mse = ((a-b)**2).mean()
    rms_a, rms_b = a.square().mean().sqrt(), b.square().mean().sqrt()
    cosine = (a*b).sum() / (a.norm()*b.norm()).clamp_min(1e-20)
    scale = (a*b).sum() / b.square().sum().clamp_min(1e-20)
    projected = scale*b
    sisdr = 10*torch.log10(projected.square().sum().clamp_min(1e-20)/(a-projected).square().sum().clamp_min(1e-20))
    return {'rms': float(rms_a), 'target_rms': float(rms_b), 'rms_ratio': float(rms_a/rms_b.clamp_min(1e-20)),
            'snr_db': float(10*torch.log10(b.square().mean().clamp_min(1e-20)/mse.clamp_min(1e-20))),
            'si_sdr_db': float(sisdr), 'zero_lag_cosine': float(cosine), 'peak': float(a.abs().max())}

for index, row in enumerate(selected):
    lookup, info = entries[row.source_id]
    p = base / 'target-cache-500h/targets' / (lookup+'.pt')
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == info['file_sha256']
    payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    record = UtteranceCache(payload['latents'], payload['teacher_audio'], payload['reference16k'], payload['metadata'], payload['cache_key'])
    record.validate()
    assert record.metadata['identity']['source']['source_id'] == row.source_id
    length = record.input_samples
    maximum_start = max(0, (length-1366)//640)
    interior = min(64, max(0, record.latents.shape[-1]//2), maximum_start)
    for start in sorted({0, interior}):
        left = max(0, start-29)
        end = min(start+64, record.latents.shape[-1])
        valid = min((end-start)*1920, length*3-start*1920)
        if valid < 4096:
            continue
        z = record.latents[..., left:end]
        target = record.teacher_audio[..., start*1920:start*1920+valid]
        reference = record.reference16k[..., start*640:start*640+valid//3]
        item = {'source_id': row.source_id, 'dataset': row.dataset, 'language': row.language,
                'start_frame': start, 'valid_samples_48k': valid, 'models': {}}
        with torch.no_grad():
            item['teacher_as_prediction'] = numbers(criterion(target, target, reference))
            item['silence_as_prediction'] = numbers(criterion(torch.zeros_like(target), target, reference))
            predictions = {}
            for name, model in models.items():
                pred = model(z)[..., (start-left)*1920:(start-left)*1920+valid]
                predictions[name] = pred
                item['models'][name] = {'loss': numbers(criterion(pred, target, reference)), 'waveform': waveform_stats(pred,target)}
        # Diagnostic output gradients have no optimizer and cannot change weights.
        if index < 8 and start == 0:
            pred = predictions['step10000'].detach().clone().requires_grad_(True)
            terms = criterion(pred, target, reference)
            weights = {'teacher_spectral': 15., 'teacher_waveform': 1., 'reference_spectral': 45.}
            gradients = {name: torch.autograd.grad(value*terms[name], pred, retain_graph=True)[0].flatten() for name, value in weights.items()}
            item['weighted_output_gradient_l2'] = {name: float(g.norm()) for name,g in gradients.items()}
            item['output_gradient_cosines'] = {a+'__'+b: float(torch.nn.functional.cosine_similarity(gradients[a],gradients[b],dim=0)) for a,b in [('teacher_spectral','reference_spectral'),('teacher_waveform','reference_spectral'),('teacher_spectral','teacher_waveform')]}
        report['results'].append(item)
    (out/'progress.json').write_text(json.dumps({'completed_rows':index+1,'selected_rows':len(selected),'last_source':row.source_id})+'\n')
    print(json.dumps({'completed_rows':index+1,'selected_rows':len(selected),'source':row.source_id,'teacher_total':item['teacher_as_prediction']['total'],'student_total':item['models']['step10000']['loss']['total']}),flush=True)

def summary(items):
    s = {'crops':len(items)}
    for variant in ['teacher_as_prediction','silence_as_prediction','step1000','step10000']:
        losses = [x[variant] if variant.endswith('prediction') else x['models'][variant]['loss'] for x in items]
        s[variant] = {k: statistics.mean(x[k] for x in losses) for k in losses[0]}
    for variant in models:
        s[variant]['waveform_mean'] = {k:statistics.mean(x['models'][variant]['waveform'][k] for x in items) for k in items[0]['models'][variant]['waveform']}
    return s
report['summary'] = summary(report['results'])
report['summary_by_group'] = {str(k): summary([r for r in report['results'] if (r['dataset'],r['language'])==k]) for k in sorted(groups)}
report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'summary':report['summary'],'finished_utc':report['finished_utc']}),flush=True)
