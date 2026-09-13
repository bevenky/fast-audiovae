"""Bounded paced comparison using the exact screened transpose builder."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
os.environ.pop('PYTORCH_MPS_PREFER_METAL', None)
sys.path.insert(0, str(HERE.parent / 'apple-gpu-v2'))
import experiment as exp
import screen_transpose as screen
import torch
from fast_audiovae.gpu import GPUDecoder


def paced_main(args):
    out = HERE / 'transpose-paced-r1.json'
    assert not out.exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    names = ('before_compile', 'transpose_pointwise_mm')
    decoders = {name: GPUDecoder(exp.build(name), torch, {'experiment': name}, 'streaming') for name in names}
    z = exp.cases()['English'][..., :16].copy()
    ref = exp.cpu_reference(exp.ORIGINAL).run(None, {'z': z})[0]
    preparation = []
    for name, decoder in decoders.items():
        with decoder.stream() as stream:
            start = time.perf_counter()
            for i in range(4):
                stream.decode_chunk(z[..., 2*i:2*i+2])
            preparation.append(dict(arm=name, first_four_calls_seconds=time.perf_counter()-start))
    counters = lambda: {str(k): dict(v) for k,v in torch._dynamo.utils.counters.items()}
    prepared = counters()
    rows = []
    with torch.inference_mode(), torch.autocast('mps', enabled=False):
        for repetition in range(2):
            order = [(name, paced) for name in names for paced in (False, True)]
            if repetition:
                order.reverse()
            for name, paced in order:
                with decoders[name].stream() as stream:
                    torch.mps.synchronize()
                    origin = time.perf_counter_ns()
                    for i in range(8):
                        target = origin+i*80_000_000
                        if paced:
                            time.sleep(max(0, (target-time.perf_counter_ns())/1e9))
                        start = time.perf_counter_ns()
                        y = stream.decode_chunk(z[..., 2*i:2*i+2])
                        end = time.perf_counter_ns()
                        comparison = exp.comparison(y, ref[..., i*3840:(i+1)*3840])
                        assert comparison['passed'], comparison
                        rows.append(dict(arm=name, paced=paced, repetition=repetition, packet=i,
                            service_ms=(end-start)/1e6,
                            arrival_lateness_ms=(start-target)/1e6 if paced else None,
                            ready_after_arrival_ms=(end-target)/1e6 if paced else None,
                            missed_deadline=(end>target+80_000_000) if paced else None,
                            comparison=comparison))
                    assert stream.frames_decoded == 16 and stream.flush().shape == (1,1,0)
    finished = counters()
    assert prepared == finished, 'Unexpected compilation during timed calls'
    summary = []
    for name in names:
        for paced in (False, True):
            selected = [r for r in rows if r['arm']==name and r['paced']==paced]
            steady = [r for r in selected if r['packet']>0]
            mean = statistics.mean(r['service_ms'] for r in steady)
            summary.append(dict(arm=name, paced=paced, noninitial_calls=len(steady),
                mean_service_ms=mean, service_rtf=mean/80,
                max_service_ms_all=max(r['service_ms'] for r in selected),
                max_arrival_lateness_ms=max(r['arrival_lateness_ms'] for r in selected) if paced else None,
                max_ready_after_arrival_ms=max(r['ready_after_arrival_ms'] for r in selected) if paced else None,
                missed_deadlines=sum(r['missed_deadline'] for r in selected) if paced else None))
    result = dict(status='passed', arms=names, rows=rows, summary=summary, preparation=preparation,
        compile_counters_after_preparation=prepared, compile_counters_after_timing=finished,
        protocol=dict(packet_ms=80, paced_interval_ms=80, packets_per_stream=8, repetitions=2,
            host_threads=1, precision='FP32', cpu_fallback=False, fast_math=False,
            boundary='Unchanged public API; complete GPU work and CPU-ready owned waveform.',
            baseline='Original CPU ONNX at same encoded latent prefix, sample-aligned per packet.',
            limitations='Short shared-desktop timing, not sustained deadline qualification. Mean excludes initial packet; maxima/deadlines include it.'),
        max_waveform_absolute_error=max(r['comparison']['max_abs'] for r in rows),
        files={str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    original_main = exp.main
    exp.main = paced_main
    try:
        screen.main(argparse.Namespace(name='transpose-paced-r1', combined=True, cpu_only=False))
    finally:
        exp.main = original_main
