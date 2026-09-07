"""Frozen real-input screen of the Intel stage-4 region, not codec RTF."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='-1', NVIDIA_VISIBLE_DEVICES='void',
                  ROCR_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    p.add_argument('--harness', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if out.exists():
        raise ValueError('Use a new result directory')
    out.mkdir(parents=True)
    os.sched_setaffinity(0, {0, 1})
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, TensorProto
    if ort.__version__ != '1.29.0':
        raise RuntimeError('ORT 1.29.0 required')
    sys.path.insert(0, str(a.harness.resolve().parent))
    spec = importlib.util.spec_from_file_location('comparison', a.harness)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    h.np, h.ort = np, ort
    config, verified = h.read_config(a.config.resolve(), a.config_sha256)
    resolve = lambda value: h.relative_path(a.config.resolve().parent, value)
    definition = next(v for v in config['models'] if v['name'] == 'stage_mkl')
    source = resolve(definition['path'])
    if h.sha(source) != 'e188d0609795d256627b4e39b632d5c5ca064256d410899ecb05ac4eb6301bc2':
        raise ValueError('Expected the accepted Intel source graph')
    libraries = [resolve(v) for v in definition['custom_libraries']]
    cases, metadata = h.read_cases(resolve(config['audio_cases']), config['kind_contracts']['audio'])
    uid = 'hi_in_00099_1919'
    z = np.ascontiguousarray(cases[uid]['z'][..., :170])
    if z.shape != (1, 64, 170):
        raise ValueError('Explicit real latent prefix missing')
    model = onnx.load(source)
    byname = {v.name: v for v in model.graph.node}
    names = ['ncc_up_4_split_matmul_current_node', 'ncc_up_4_split_matmul_previous_unshifted_node',
             'ncc_phase_finish_140_op', 'sp_stage_c128']
    nodes = [copy.deepcopy(byname[n]) for n in names]
    current, previous, phase, stage = nodes
    xname, yname = current.input[1], stage.output[0]
    if previous.input[1] != xname or stage.input[0] != phase.output[0]:
        raise ValueError('Unexpected stage-4 region')
    prefix = copy.deepcopy(model)
    del prefix.graph.output[:]
    prefix.graph.output.append(helper.make_tensor_value_info(xname, TensorProto.FLOAT, [1, 256, 40800]))

    def session(m, extra=False):
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = 2, 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        so.add_session_config_entry('session.inter_op.allow_spinning', '0')
        for library in libraries + ([a.library] if extra else []):
            so.register_custom_ops_library(str(library))
        result = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
        result.disable_fallback()
        if result.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('CPU-only session required')
        return result

    prefix_session = session(prefix)
    x = prefix_session.run(None, {prefix_session.get_inputs()[0].name: z})[0]
    if x.shape != (1, 256, 40800) or not np.isfinite(x).all():
        raise ValueError('Unexpected real region input')
    np.save(out / 'region_input.npy', x, allow_pickle=False)
    del prefix_session, prefix
    required = {value for node in nodes for value in node.input}
    constants = [copy.deepcopy(v) for v in model.graph.initializer if v.name in required]
    ref = helper.make_model(helper.make_graph(nodes, 'accepted stage4',
        [helper.make_tensor_value_info(xname, TensorProto.FLOAT, [1, 256, 'T'])],
        [helper.make_tensor_value_info(yname, TensorProto.FLOAT, [1, 128, 'T2'])], constants),
        opset_imports=list(model.opset_import), ir_version=model.ir_version)
    del model
    onnx.save(ref, out / 'region_reference.onnx')
    inputs = [xname, current.input[0], previous.input[0], phase.input[2], *stage.input[1:]]
    if len(inputs) != 28:
        raise ValueError('Unexpected 28-input region contract')

    def candidate(q, mode, debug=False):
        m = copy.deepcopy(ref)
        del m.graph.node[:]
        outputs = [yname]
        if debug:
            outputs = [current.output[0], previous.output[0], phase.output[0], 'u1', 'u2', yname]
            del m.graph.output[:]
            for i, value in enumerate(outputs):
                dims = [1, 256, 'T'] if i < 2 else [1, 128, 'T2']
                m.graph.output.append(helper.make_tensor_value_info(value, TensorProto.FLOAT, dims))
        m.graph.node.append(helper.make_node('UpsampleStageDebugF32' if debug else 'UpsampleStageF32',
            inputs, outputs, name='up_stage4', domain='fast.audiovae.upsample.experimental',
            native_abi=1, channels=128, stride=2, tile_time=q, segments=2, backend=5,
            matrix_mode=0, matrix_isa=512, projection_mode=mode, projection_isa=0 if mode else 512))
        m.opset_import.append(helper.make_opsetid('fast.audiovae.upsample.experimental', 1))
        return m

    ref_debug = copy.deepcopy(ref)
    ref_debug.graph.node[-1].op_type = 'StageStackDebugF32'
    del ref_debug.graph.node[-1].output[:]
    ref_debug.graph.node[-1].output.extend(['u1', 'u2', yname])
    del ref_debug.graph.output[:]
    for i, name in enumerate([current.output[0], previous.output[0], phase.output[0], 'u1', 'u2', yname]):
        ref_debug.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT,
            [1, 256, 'T'] if i < 2 else [1, 128, 'T2']))
    reference_debug_session = session(ref_debug)
    expected = reference_debug_session.run(None, {xname: x})
    del reference_debug_session
    result = dict(status='running', scope='Isolated real stage4 screen only; no full decoder RTF',
        gpu_used=False, runtime=ort.__version__, onnx=onnx.__version__, threads=2, affinity=[0, 1],
        source_model_sha256=h.sha(source), source_config_sha256=a.config_sha256,
        screen_script_sha256=h.sha(__file__), harness_sha256=h.sha(a.harness),
        reference_graph_sha256=h.sha(out / 'region_reference.onnx'), candidate_graph_sha256={},
        library_sha256=h.sha(a.library), native_libraries=[{'path':str(v), 'sha256':h.sha(v)} for v in libraries],
        source_uid=uid, latent_frames=170, generated_audio_seconds=6.8,
        captured_input_sha256=h.sha(out/'region_input.npy'), verified_artifact_sha256=verified,
        protocol={'warmups':2, 'repeats':7, 'random_seed':9921, 'only_production_final_output_timed':True,
                  'unchanged_quality_gate':'atol1e-5 rtol1e-4', 'selection':'largest accepted median region gain; full decoder validation required'},
        checks={}, production_warmup_checks={}, observations=[], summary={})
    def save():
        (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    sessions = {'reference':session(ref)}
    save()
    for mode in (0, 1):
        for q in (64, 128, 256):
            name = f'p{mode}_q{q}'
            debug = session(candidate(q, mode, True), True)
            actual = debug.run(None, {xname:x})
            checks = [h.compare(v, r) for v, r in zip(actual, expected)]
            result['checks'][name] = checks
            save()
            if not all(v['passed'] for v in checks):
                raise RuntimeError('Real intermediate output gate failed: ' + name)
            del debug, actual
            m = candidate(q, mode)
            onnx.save(m, out / (name + '.onnx'))
            result['candidate_graph_sha256'][name] = h.sha(out / (name + '.onnx'))
            sessions[name] = session(m, True)
    expected_final = expected[-1]
    del expected
    for name, s in sessions.items():
        result['production_warmup_checks'][name] = []
        for _ in range(2):
            y = s.run(None, {xname:x})[0]
            gate = h.compare(y, expected_final)
            result['production_warmup_checks'][name].append(gate)
            if not gate['passed']:
                save()
                raise RuntimeError('Real production waveform gate failed: ' + name)
    del expected_final
    save()
    before = x.tobytes()
    result['host_timing_before'] = h.host_snapshot()
    result['gpu_library_mappings_before'] = h.gpu_library_mappings()
    rng = random.Random(9921)
    for repeat in range(7):
        order = list(sessions)
        rng.shuffle(order)
        for name in order:
            start = time.perf_counter_ns()
            y = sessions[name].run(None, {xname:x})[0]
            elapsed = (time.perf_counter_ns() - start) / 1e6
            result['observations'].append({'repeat':repeat, 'candidate':name, 'milliseconds':elapsed})
        save()
    if before != x.tobytes():
        raise RuntimeError('Input mutation')
    result['host_timing_after'] = h.host_snapshot()
    result['host_timing_delta'] = h.host_delta(result['host_timing_before'], result['host_timing_after'])
    result['gpu_library_mappings_after'] = h.gpu_library_mappings()
    for name in sessions:
        values = [v['milliseconds'] for v in result['observations'] if v['candidate']==name]
        result['summary'][name] = {'median_ms':statistics.median(values), 'mean_ms':statistics.mean(values), 'samples_ms':values}
    baseline = result['summary']['reference']['median_ms']
    for value in result['summary'].values():
        value['reduction_percent_vs_reference_median'] = 100 * (1-value['median_ms']/baseline)
    result['status']='complete'
    save()
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':
    main()
