"""Retained Apple FP32 and stock full-waveform validation and FLOAT export only. No INT8, Mimi or timing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--harness', required=True, type=Path)
    p.add_argument('--harness-sha256', required=True)
    p.add_argument('--platform', required=True, choices=('amd', 'apple'))
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--affinity', help='Comma-separated Linux CPU IDs; required on AMD')
    p.add_argument('--output-dir', required=True, type=Path)
    a = p.parse_args()
    require(sha(a.harness) == a.harness_sha256, 'Harness hash mismatch')
    spec = importlib.util.spec_from_file_location('public_decoder_harness', a.harness)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    environment = h.require_cpu_environment()
    require(1 <= a.threads <= 64, 'Invalid thread count')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        require(os.environ.get(key) == '1', key+' must be 1 before numerical imports')
    if a.platform == 'amd':
        require(platform.system() == 'Linux' and 'AuthenticAMD' in Path('/proc/cpuinfo').read_text(), 'AMD Linux host required')
        require(a.affinity, 'Explicit AMD affinity required')
        cpus = sorted({int(x) for x in a.affinity.split(',')})
        require(len(cpus) == a.threads and set(cpus) <= os.sched_getaffinity(0), 'Invalid AMD affinity')
        os.sched_setaffinity(0, cpus)
        require(sorted(os.sched_getaffinity(0)) == cpus, 'AMD affinity not applied')
    else:
        require(platform.system() == 'Darwin' and platform.machine() in ('arm64', 'aarch64'), 'Native Apple ARM required')
        require(a.affinity is None, 'No unsupported macOS CPU affinity claim')
        cpus = None
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    require(ort.__version__ == '1.29.0', 'ONNX Runtime 1.29.0 required')
    h.np, h.ort = np, ort
    require(a.platform == 'apple', 'This derivative is reserved for Apple validation')
    import onnx
    require(onnx.__version__ == '1.22.0', 'ONNX parser1.22.0 required')
    cfg, hashes = h.read_config(a.config, a.config_sha256)
    names = [m['name'] for m in cfg['models']]
    require(set(names) == {'audio_stock', 'fast_fp32'}, 'Two retained FP32 models required')
    require(len(cfg['expected_uids']) == 60 and len(cfg['timing_uids']) == 10, 'Frozen 60/10 corpus required')
    require(not any(m.get('approximate',False) for m in cfg['models']), 'Only retained FP32 models permitted')
    resolve = lambda value: (a.config.parent/value).resolve()
    inputs = {kind: h.read_cases(resolve(cfg[kind+'_cases']), cfg['kind_contracts'][kind])[0]
              for kind in {m['kind'] for m in cfg['models']}}
    require(all(set(data) == set(cfg['expected_uids']) for data in inputs.values()), 'Corpus UID mismatch')
    timed_uids = []
    validate_uids = cfg['expected_uids']
    require(not a.output_dir.exists(), 'Fresh output directory required')
    a.output_dir.mkdir(parents=True)
    record = {'status': 'running', 'platform': a.platform, 'mode': 'validation_only',
              'gpu_used': False, 'threads': a.threads, 'affinity': cpus, 'environment': environment,
              'runtime': {'onnx': onnx.__version__, 'soundfile': sf.__version__, 'onnxruntime': ort.__version__, 'build': ort.get_build_info(), 'numpy': np.__version__,
                          'python': platform.python_version(), 'system': platform.platform()},
              'config_sha256': a.config_sha256, 'harness_sha256': a.harness_sha256,
              'script_sha256': sha(__file__), 'derived_from_runner_sha256': '04c49ba9c6f8ccfc9bd001cecb7193d6cf348edb73ab4b389918d0b1031c8882', 'artifact_sha256': hashes,
              'checks': [], 'exports': [], 'measurements': [], 'providers': [], 'failures': [],
              'protocol': {'validation_uids': validate_uids, 'timing_uids': timed_uids, 'warmups': 0, 'repeats': 0,
                           'seed': 20260908, 'decoder_only': True, 'same_time_column_scales': True,
                           'fp32_gate': 'atol1e-5/rtol1e-4 against the frozen reference and fresh stock',
                           'int8_gate': 'Not applicable; no INT8 execution',
                           'timing': 'None. Full waveform validation and export only'},
              'host_before': h.host_snapshot(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    path = a.output_dir/'results.json'
    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temp.replace(path)
    def pulse(phase, **fields):
        record['progress'] = {'phase': phase, **fields}
        save()
        print(json.dumps(record['progress']), flush=True)
    def bit_equal(x, y):
        return x.shape == y.shape and np.array_equal(x.view(np.uint32), y.view(np.uint32))
    def wave_hash(x):
        return hashlib.sha256(x.tobytes()).hexdigest()
    def gpu_inventory():
        result = h.gpu_library_mappings()
        if platform.system() == 'Darwin':
            lib = ctypes.CDLL(None)
            lib._dyld_image_count.restype = ctypes.c_uint32
            lib._dyld_get_image_name.argtypes = [ctypes.c_uint32]
            lib._dyld_get_image_name.restype = ctypes.c_char_p
            images = [lib._dyld_get_image_name(i).decode() for i in range(lib._dyld_image_count())]
            result = {'available': True, 'source': 'dyld loaded images',
                      'basenames': sorted({Path(v).name for v in images if any(s in v.lower()
                          for s in ('libcuda', 'libcublas', 'libcudnn', 'providers_cuda', 'providers_tensorrt', 'librocblas', 'libamdhip'))}),
                      'metal_images': sorted({Path(v).name for v in images if '/Metal.framework/' in v}),
                      'scope': 'Loaded libraries inventory; sessions explicitly use CPU provider and native kernels use CPU instructions'}
        require(result['available'] and not result['basenames'], 'Unexpected GPU compute runtime mapping')
        return result
    sessions, stock, validated = {}, {}, {}
    def run(model, z):
        session, input_name = sessions[model['name']]
        y = session.run(None, {input_name: z})[0]
        contract = cfg['kind_contracts'][model['kind']]
        require(y.dtype == np.float32 and y.shape == (1, 1, z.shape[-1]*contract['hop']) and np.isfinite(y).all(), 'Invalid waveform')
        return y
    try:
        record['gpu_mappings_before'] = gpu_inventory()
        for model in cfg['models']:
            name, kind = model['name'], model['kind']
            pulse('session', model=name)
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = a.threads, 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.add_session_config_entry('session.intra_op.allow_spinning', '0')
            so.add_session_config_entry('session.inter_op.allow_spinning', '0')
            libraries = model.get('custom_libraries', []) or ([model['custom_library']] if model.get('custom_library') else [])
            for library in libraries:
                so.register_custom_ops_library(str(resolve(library)))
            session = ort.InferenceSession(str(resolve(model['path'])), sess_options=so, providers=['CPUExecutionProvider'])
            session.disable_fallback()
            require(session.get_providers() == ['CPUExecutionProvider'], 'Unexpected execution provider')
            require(len(session.get_inputs()) == len(session.get_outputs()) == 1, 'Single input/output required')
            require(session.get_inputs()[0].type == session.get_outputs()[0].type == 'tensor(float)', 'FP32 interface required')
            sessions[name] = (session, session.get_inputs()[0].name)
            record['providers'].append({'model': name, 'providers': session.get_providers()})
            for uid in validate_uids:
                case = inputs[kind][uid]
                z, ref = case['z'], case['ref']
                before = wave_hash(z)
                y = run(model, z)
                require(wave_hash(z) == before, 'Latents mutated')
                if name == 'audio_stock':
                    stock[uid] = y.copy()
                if not model.get('approximate', False):
                    require(h.compare(y, ref)['passed'], name+' failed saved FP32 reference')
                    if kind == 'audio':
                        require(h.compare(y, stock[uid])['passed'], name+' failed fresh stock parity')
                validated[name, uid] = wave_hash(y)
                record['checks'].append({'model': name, 'uid': uid, 'test': 'full_waveform', 'passed': True,
                                         'latent_sha256': before, 'waveform_sha256': validated[name, uid],
                                         'fp32_parity_required': not model.get('approximate', False)})
                folder = a.output_dir/'wavs'/name
                folder.mkdir(parents=True, exist_ok=True)
                wav = folder/(uid+'.wav')
                rate = cfg['kind_contracts'][kind]['sample_rate']
                sf.write(wav, y.ravel(), rate, subtype='FLOAT')
                restored, restored_rate = sf.read(wav, dtype='float32')
                require(restored_rate == rate and bit_equal(restored, y.ravel()), 'WAV serialization changed samples')
                record['exports'].append({'model': name, 'uid': uid, 'path': str(wav.resolve()), 'sha256': sha(wav),
                                          'rate': rate, 'samples': len(restored), 'subtype': 'FLOAT'})
            z = np.ascontiguousarray(inputs[kind][cfg['timing_uids'][0]]['z'][..., :65])
            hop = cfg['kind_contracts'][kind]['hop']
            base = run(model, z)
            require(bit_equal(base, run(model, z)), name+' repeat failed')
            for cut in (1, 8, 32, 64):
                future = z.copy()
                future[..., cut:] += np.float32(.37)
                require(bit_equal(base[..., :cut*hop], run(model, future)[..., :cut*hop]), name+' future invariance failed')
                record['checks'].append({'model': name, 'test': 'future', 'cut': cut, 'passed': True})
            for length in (1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64):
                small = run(model, np.ascontiguousarray(z[..., :length]))
                require(h.compare(small, base[..., :length*hop])['passed'], name+' short prefix failed at '+str(length))
                record['checks'].append({'model': name, 'test': 'short_prefix', 'length': length, 'passed': True})
            require(bit_equal(base, run(model, z)), name+' long-short-long failed')
            small_z = np.ascontiguousarray(z[..., :33])
            small_reference = run(model, small_z)
            with ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(run, model, z)
                f2 = pool.submit(run, model, small_z)
                require(bit_equal(f1.result(), base) and bit_equal(f2.result(), small_reference), name+' concurrent replay failed')
            record['checks'].append({'model': name, 'test': 'repeat_history_concurrency', 'passed': True})
            pulse('validated', model=name)
        require(len(record['exports']) == 120, 'Expected both FP32 models and 60 FLOAT WAV exports each')
        require(h.read_config(a.config, a.config_sha256)[1] == hashes and sha(a.harness) == a.harness_sha256
                and sha(__file__) == record['script_sha256'], 'Input artifact changed')
        record['gpu_mappings_after'] = gpu_inventory()
        record['status'] = 'complete'
        record['quality_pass'] = None
        record['commit_eligible'] = False
        record['limit'] = 'Completed runtime gates and waveform exports only. No warmup or timing; perceptual quality and adoption remain separate.'
    except Exception as exc:
        record['status'] = 'failed'
        record['failures'].append({'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        record['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save()
    print(json.dumps({'status': record['status'], 'checks': len(record['checks']), 'summary': record.get('summary')}), flush=True)


if __name__ == '__main__':
    main()
