"""Reproduce the exact historical teacher batch without modifying cached pairs."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import fcntl, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from diagnostic_common import load_context, atomic_json, status
from repair_natural_history import source_rows, authentic_audio, raw_metrics, tensor_hash
from audiovae_student.objective_comparison import state_fingerprint

OUT = Path('/tmp/fast-audiovae-recovery-20260909/cache')
IDS = ['kannada:6192449488033510_chunk_1.flac', 'freesound:386521',
       'urdu:5910974511198452_chunk_1.flac', 'jvnv:M2_surprise_regular_56.wav',
       'freesound:119459', 'urdu:5910974511173186_chunk_1.flac',
       'bengali:1407374883904482_chunk_1.flac', '2277-149896-0009']
LENGTHS = [83984, 88000, 88288, 88589, 88697, 88832, 90848, 93040]

def metric(a, b):
    return raw_metrics(a.detach().cpu(), b.detach().cpu())

@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'historical-batch.json').exists():
        raise RuntimeError('Refusing to overwrite completed experiment')
    ctx = load_context()
    rows, pins = source_rows(ctx)
    teacher = ctx.teacher()
    state_before = state_fingerprint(teacher.model.state_dict())
    audios, provenance = zip(*(authentic_audio(rows[sid]) for sid in IDS))
    if [a.shape[-1] for a in audios] != LENGTHS:
        raise RuntimeError('Historical batch lengths differ')
    batch = torch.cat([F.pad(a, (0, 93440-a.shape[-1])) for a in audios]).to(teacher.device)
    batches = []
    for i in range(3):
        status('exact_historical_batch', repeat=i)
        batches.append(teacher.encode(batch).detach().cpu())
    bz = batches[-1]
    result = {'format_version': 1, 'batch_shape': list(batch.shape),
              'batch_input_sha256': tensor_hash(batch), 'source_manifest_pins': pins,
              'teacher_state_sha256': state_before, 'teacher': teacher.provenance,
              'runtime': {'torch': str(torch.__version__), 'cudnn': torch.backends.cudnn.version(),
                          'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                          'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                          'deterministic': torch.are_deterministic_algorithms_enabled()},
              'batch_repeats': [metric(z, bz) for z in batches], 'cases': []}
    serial = {}
    for i, (sid, audio, info) in enumerate(zip(IDS, audios, provenance)):
        status('serial_vs_historical_batch', source=sid)
        sz = teacher.encode(audio.to(teacher.device)).detach().cpu()
        repeat = teacher.encode(audio.to(teacher.device)).detach().cpu()
        pz = teacher.encode(batch[i:i+1].contiguous()).detach().cpu()
        serial[sid] = sz
        cached = next((c for c in ctx.heldout if c.source_id == sid and c.context_start_frame == 0), None)
        n = sz.shape[-1]
        row = {'source_id': sid, 'source': info,
               'serial_repeat': metric(sz, repeat),
               'batch_vs_serial': metric(bz[i:i+1, :, :n], sz),
               'padded_single_vs_serial': metric(pz[..., :n], sz),
               'batch_vs_padded_single': metric(bz[i:i+1], pz)}
        if cached is not None:
            f = cached.latents.shape[-1]
            row['cached_frames'] = f
            row['serial_vs_cache'] = metric(sz[..., :f], cached.latents)
            row['batch_vs_cache'] = metric(bz[i:i+1, :, :f], cached.latents)
        result['cases'].append(row)
        atomic_json(OUT/'historical-batch-progress.json', result)
    reverse = teacher.encode(batch.flip(0)).detach().cpu().flip(0)
    result['batch_order_change'] = metric(reverse, bz)
    result['teacher_state_after_sha256'] = state_fingerprint(teacher.model.state_dict())
    if result['teacher_state_after_sha256'] != state_before:
        raise RuntimeError('Frozen teacher state changed')
    result['checkpoint_preservation'] = ctx.verify_files()
    result['parameter_updates'] = 0
    atomic_json(OUT/'historical-batch.json', result)
    torch.save({'batch_latents': bz, 'serial_latents': serial, 'row_ids': IDS}, OUT/'probe-latents.pt')
    status('cache_batch_probe_complete', result=str(OUT/'historical-batch.json'))

if __name__ == '__main__':
    lock = Path('/workspace/fast-audiovae-convnext-20260909-r9/training-runs/.decoder-recipe-v2-expressive.runner.lock')
    with lock.open('rb') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
