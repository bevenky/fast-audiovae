"""Short public-API arrival/deadline probe; per-thread QoS restored afterward."""
from pathlib import Path
import ctypes as c
import hashlib
import json
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'apple-gpu-v2'))
import experiment as exp
import torch
from fast_audiovae.gpu import GPUDecoder


def main():
    out = HERE / 'queue-qos-r1.json'
    assert not out.exists()
    lib = c.CDLL('/usr/lib/libSystem.B.dylib')
    lib.pthread_self.restype = c.c_void_p
    lib.pthread_get_qos_class_np.argtypes = [c.c_void_p, c.POINTER(c.c_uint), c.POINTER(c.c_int)]
    lib.pthread_get_qos_class_np.restype = c.c_int
    lib.pthread_set_qos_class_self_np.argtypes = [c.c_uint, c.c_int]
    lib.pthread_set_qos_class_self_np.restype = c.c_int

    def get():
        qos, priority = c.c_uint(), c.c_int()
        assert lib.pthread_get_qos_class_np(lib.pthread_self(), c.byref(qos), c.byref(priority)) == 0
        return qos.value, priority.value

    def set_qos(value):
        assert lib.pthread_set_qos_class_self_np(*value) == 0
        assert get() == value

    initial = get()
    initiated = (0x19, 0)
    modes = [('initial', initial)]
    # UNSPECIFIED cannot be restored with the setter; do not override it.
    if initial[0] in (0x09, 0x11, 0x15, 0x19, 0x21) and initial != initiated:
        modes.append(('user_initiated', initiated))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = exp.build('before_compile')
    decoder = GPUDecoder(model, torch, {'experiment': 'queue-qos'}, 'streaming')
    z = exp.cases()['English']
    with decoder.stream() as stream:
        for i in range(4):
            stream.decode_chunk(z[..., i*2:i*2+2].copy())
    rows = []
    try:
        with torch.inference_mode(), torch.autocast('mps', enabled=False):
            for repetition in range(2):
                order = [(name, qos, paced) for name, qos in modes for paced in (False, True)]
                if repetition:
                    order.reverse()
                for name, qos, paced in order:
                    if get() != qos:
                        set_qos(qos)
                    with decoder.stream() as stream:
                        torch.mps.synchronize()
                        origin = time.perf_counter_ns()
                        for i in range(8):
                            target = origin + i * 80_000_000
                            if paced:
                                time.sleep(max(0, (target-time.perf_counter_ns()) / 1e9))
                            begin = time.perf_counter_ns()
                            y = stream.decode_chunk(z[..., i*2:i*2+2])
                            end = time.perf_counter_ns()
                            rows.append(dict(repetition=repetition, mode=name, qos=get(), paced=paced,
                                packet=i, service_ms=(end-begin)/1e6,
                                arrival_lateness_ms=(begin-target)/1e6 if paced else None,
                                ready_after_arrival_ms=(end-target)/1e6 if paced else None,
                                missed_deadline=(end>target+80_000_000) if paced else None,
                                output_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
    finally:
        if get() != initial:
            set_qos(initial)
    for i in range(8):
        assert len({r['output_sha256'] for r in rows if r['packet']==i}) == 1
    summary = []
    for name, qos in modes:
        for paced in (False, True):
            selected = [r for r in rows if r['mode']==name and r['paced']==paced]
            steady = [r for r in selected if r['packet']>0]
            summary.append(dict(mode=name, qos=qos, paced=paced, steady_packets=len(steady),
                mean_service_ms=statistics.mean(r['service_ms'] for r in steady),
                max_service_ms_all=max(r['service_ms'] for r in selected),
                max_arrival_lateness_ms=max(r['arrival_lateness_ms'] for r in selected) if paced else None,
                max_ready_after_arrival_ms=max(r['ready_after_arrival_ms'] for r in selected) if paced else None,
                missed_deadlines=sum(r['missed_deadline'] for r in selected) if paced else None))
    result = dict(status='passed', initial_qos=initial, final_qos=get(), modes=modes, rows=rows,
        summary=summary, packet_ms=80, exact_outputs_equal=True, torch=torch.__version__,
        limitations='Short shared-desktop probe, not sustained deadline qualification; QoS affects caller scheduling, not a direct GPU frequency setting.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
