"""One bounded CPU-only regional screen of the GPU paired-projection formula.

No production rewrite. Run only after the parent grants the native CPU lease.
The control retains projected history; the candidate retains activated history.
Both include all recurring packing, layout, bias, allocation and state work.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(ORT_DISABLE_TELEMETRY='1', CUDA_VISIBLE_DEVICES='-1')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = ROOT/'outputs/apple-streaming-promotion-v1/config-r3.json'
SELECTED = '5918e523939aba3a6f72e32b88b27a0a1f39828a376b0ed748a7cb43e70ce841'
NAMES = ('ncc_up_1_split_matmul_current_node_fixed_right',
         'ncc_up_1_split_matmul_previous_unshifted_node_fixed_right')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Preserve existing results')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(version='apple_gpu_formula_cpu_applicability_v1', status='starting',
                  calls=0, native_seconds=0., math=[], timing=[], summary=[])
    save = lambda: output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save()
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper as h, numpy_helper as nh
        require(platform.system() == 'Darwin' and platform.machine() == 'arm64'
                and ort.__version__ == '1.30.0', 'Apple arm64 / ORT1.30 required')
        config = json.loads(CONFIG.read_text()); bundle = Path(config['bundle'])
        require(sha(bundle/'bundle.json') == config['bundle_manifest_sha256'], 'Baseline manifest changed')
        manifest = json.loads((bundle/'bundle.json').read_text())
        native = manifest['native']['Darwin/arm64']
        spec = manifest['streaming']['models'][native['model']]
        require(spec['model_sha256'] == SELECTED == config['stream_graph_sha256'], 'Wrong CPU baseline')
        files = {str(Path(__file__).resolve()): sha(__file__), str(CONFIG): sha(CONFIG),
                 str(bundle/'bundle.json'): config['bundle_manifest_sha256'],
                 str(bundle/spec['model']): SELECTED,
                 str(bundle/native['library']): native['library_sha256']}
        libraries = [dict(library=native['library'], sha256=native['library_sha256']),
                     *spec['additional_libraries']]
        for entry in [*libraries, *spec['dependencies']]:
            files[str(bundle/entry['library'])] = entry['sha256']
        require(all(sha(p) == s for p, s in files.items()), 'Source/runtime identity mismatch before load')
        graph = onnx.load(bundle/spec['model'])
        nodes = {n.name:n for n in graph.graph.node}
        current, previous = [copy.deepcopy(nodes[n]) for n in NAMES]
        require(current.input[0] == previous.input[0], 'Control projection input differs')
        require(current.domain == 'fast.audiovae.apple.libxsmm.panel.v1'
                and previous.domain == 'fast.audiovae.apple.multitile.v1', 'Wrong native projection routes')
        phase = [copy.deepcopy(n) for n in graph.graph.node
                 if n.op_type == 'StatefulPhaseFinishF32'
                 and list(n.input[:2]) == [current.output[0], previous.output[0]]]
        require(len(phase) == 1, 'First phase region is not unique'); phase = phase[0]
        initializer = {w.name:w for w in graph.graph.initializer}
        weights = [np.array(nh.to_array(initializer[n.input[1]]), copy=True) for n in (current, previous)]
        require(all(w.shape == (8192,2048) and w.dtype == np.float32 and np.isfinite(w).all()
                    for w in weights), 'Exact original first-pair FP32 weights required')
        bias = np.array(nh.to_array(initializer[phase.input[3]]), copy=True)
        require(bias.shape == (1024,), 'Wrong phase bias')
        paired = np.concatenate(weights, axis=1)
        constant_control = [copy.deepcopy(initializer[n]) for n in
                            (current.input[1], previous.input[1], phase.input[3])]
        opsets = [copy.deepcopy(o) for o in graph.opset_import]
        ir_version = graph.ir_version
        del graph, initializer, nodes
        plan = dict(source_sha256=files[str(Path(__file__).resolve())], graph_sha256=SELECTED,
            control_nodes=list(NAMES)+[phase.name], threads=[1,4], frames=[1,2],
            dtype='FP32', provider='CPUExecutionProvider', warmups=2, paired_repeats=6,
            pair_order='Three AB and three BA orders, shuffled deterministically per geometry',
            mathematical_scope='Actual first-upsample weights; deterministic synthetic signed/quiet/zero activations',
            control='Frozen native LIBXSMM panel64 + Kleidi multitile + native phase/state finish',
            candidate='GPU stage0 algebra: one standard ORT W[8192,4096] @ features[1,4096,T]',
            state_difference='Control [1,8192,1] previous projection; candidate [1,2048,1] activated input.',
            state_validation='Independent histories carried through signed, quiet and zero packets; compare every output; candidate next state equals final input.',
            timing_boundary='Whole session.run, including concat, matmul, transpose, bias, output and next-state allocation',
            excluded='One-time literal weight preparation, session construction, numerical assertions and hashes',
            limits=dict(exact_calls=88, completed_session_run_seconds=2.0),
            production_change=False, gpu_used=False,
            limitation='Regional backend test; no full decoder, retained state-schema compatibility, arbitrary packet or perceptual qualification')
        plan_path = output.with_suffix('.plan.json')
        with plan_path.open('x') as f: json.dump(plan, f, indent=2); f.write('\n')
        result.update(status='running', plan=plan, plan_sha256=sha(plan_path), files=files,
                      runtime=dict(onnxruntime=ort.__version__, runtime_file=ort.__file__,
                                   machine=platform.machine(), libraries=libraries),
                      weight_sha256=[hashlib.sha256(w.tobytes()).hexdigest() for w in weights])
        save()

        def make_model(t, candidate):
            value = lambda n,s: h.make_tensor_value_info(n, TensorProto.FLOAT, s)
            if not candidate:
                g = h.make_graph([current,previous,phase], 'frozen_first_region',
                    [value(current.input[0], [1,2048,t]), value(phase.input[2], [1,8192,1])],
                    [value(phase.output[0], [1,1024,t*8]), value(phase.output[1], [1,8192,1])],
                    constant_control)
                m = h.make_model(g, opset_imports=opsets)
            else:
                const = [nh.from_array(paired, 'paired_weight'), nh.from_array(bias[None,:,None], 'bias')]
                for name, v in {'starts':[0], 'ends':[t-1], 'axes':[2], 'steps':[1],
                                'last_start':[t-1], 'last_end':[t],
                                'phase_shape':[1,1024,8,t], 'audio_shape':[1,1024,t*8]}.items():
                    const.append(nh.from_array(np.array(v,np.int64), name))
                ns = [h.make_node('Slice',['x','starts','ends','axes','steps'],['earlier']),
                      h.make_node('Concat',['history','earlier'],['previous_x'],axis=2),
                      h.make_node('Concat',['x','previous_x'],['features'],axis=1),
                      h.make_node('MatMul',['paired_weight','features'],['projected']),
                      h.make_node('Reshape',['projected','phase_shape'],['phases']),
                      h.make_node('Transpose',['phases'],['times'],perm=[0,1,3,2]),
                      h.make_node('Reshape',['times','audio_shape'],['unbiased']),
                      h.make_node('Add',['unbiased','bias'],['y']),
                      h.make_node('Slice',['x','last_start','last_end','axes','steps'],['next_history'])]
                g = h.make_graph(ns, 'gpu_paired_formula_cpu',
                    [value('x',[1,2048,t]),value('history',[1,2048,1])],
                    [value('y',[1,1024,t*8]),value('next_history',[1,2048,1])],const)
                m = h.make_model(g, opset_imports=[h.make_opsetid('',17)])
            m.ir_version = min(ir_version,10)
            onnx.checker.check_model(m)
            return m

        def session(m, threads, control):
            options = ort.SessionOptions(); options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1; options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.log_severity_level = 3
            for key in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):
                options.add_session_config_entry(key,'0')
            if control:
                for entry in libraries: options.register_custom_ops_library(str(bundle/entry['library']))
            s = ort.InferenceSession(m.SerializeToString(),options,providers=['CPUExecutionProvider'])
            s.disable_fallback(); require(s.get_providers() == ['CPUExecutionProvider'], 'Wrong active provider')
            return s

        def call(s, feeds):
            require(result['calls'] < 88 and result['native_seconds'] < 2., 'Native budget exhausted')
            begin = time.perf_counter()
            try: return s.run(None, feeds)
            finally:
                elapsed = time.perf_counter()-begin
                result['calls'] += 1; result['native_seconds'] += elapsed
                require(result['native_seconds'] <= 2., 'Native budget exceeded; no further calls')

        for threads in (1,4):
            for t in (1,2):
                gen = np.random.default_rng(6107+t)
                base = session(make_model(t,False),threads,True)
                candidate = session(make_model(t,True),threads,False)
                hs = [np.zeros((1,8192,1),np.float32),np.zeros((1,2048,1),np.float32)]
                fixed_states = None
                for pattern, scale in [('signed',.03),('quiet',3e-6),('zero',0.)]:
                    x = (gen.standard_normal((1,2048,t))*scale).astype(np.float32)
                    before_inputs = [v.tobytes() for v in (x,*hs)]
                    outputs = [call(base,{current.input[0]:x,phase.input[2]:hs[0]}),
                               call(candidate,{'x':x,'history':hs[1]})]
                    a,b = [p[0] for p in outputs]
                    passed = bool(np.isfinite(a).all() and np.isfinite(b).all()
                                  and a.shape == b.shape == (1,1024,t*8)
                                  and np.allclose(a,b,atol=1e-5,rtol=1e-4))
                    result['math'].append(dict(threads=threads,frames=t,pattern=pattern,passed=passed,
                        max_abs=float(np.abs(a.astype(np.float64)-b).max())))
                    require(passed, 'Paired CPU formula failed unchanged numerical tolerance')
                    require(np.array_equal(outputs[1][1],x[...,-1:])
                            and outputs[0][1].shape == (1,8192,1)
                            and np.isfinite(outputs[0][1]).all(), 'Wrong next-state contract')
                    require([v.tobytes() for v in (x,*hs)] == before_inputs
                            and all(not np.shares_memory(p[1],x) and not np.shares_memory(p[1],old)
                                    for p,old in zip(outputs,hs)), 'Input mutation or unowned next state')
                    hs = [p[1] for p in outputs]
                    if fixed_states is None: fixed_states = [v.copy() for v in hs]
                x = (gen.standard_normal((1,2048,t))*.03).astype(np.float32)
                feeds = [{current.input[0]:x,phase.input[2]:fixed_states[0]},
                         {'x':x,'history':fixed_states[1]}]
                sessions = [base,candidate]
                for _ in range(2):
                    for s,f in zip(sessions,feeds): call(s,f)
                orders = [(0,1)]*3+[(1,0)]*3; random.Random(6100+threads*10+t).shuffle(orders)
                reductions=[]
                for repeat,order in enumerate(orders):
                    seconds=[0.,0.]
                    for index in order:
                        before=result['native_seconds']; call(sessions[index],feeds[index])
                        seconds[index]=result['native_seconds']-before
                    reduction=100*(1-seconds[1]/seconds[0]); reductions.append(reduction)
                    result['timing'].append(dict(threads=threads,frames=t,repeat=repeat,order=list(order),
                        baseline_seconds=seconds[0],candidate_seconds=seconds[1],reduction_percent=reduction))
                result['summary'].append(dict(threads=threads,packet_ms=t*40,pairs=6,
                    median_reduction_percent=statistics.median(reductions),wins=sum(r>0 for r in reductions),
                    locally_qualifies=statistics.median(reductions)>=10 and all(r>0 for r in reductions)))
                save(); del base,candidate,sessions; gc.collect()
        require(result['calls'] == 88, 'Wrong fixed call accounting')
        require(all(sha(p) == s for p,s in files.items()), 'Source/runtime changed during screen')
        result.update(status='passed',artifacts_unchanged=True)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc)); raise
    finally: save()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'cpu-applicability-r1.json')
    main(parser.parse_args().output)
