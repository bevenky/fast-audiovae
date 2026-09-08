"""Load the validated AudioVAE2 decoder using CPU execution exclusively.

Select a platform-specific native library and graph when compatible. The
native library detects supported CPU instructions. Other platforms use the
standard-ONNX fallback. Streaming sessions use explicit per-stream history.
"""
import ctypes
import json
import os
import platform
from pathlib import Path

import onnxruntime as ort


def _default_threads():
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count()
    return min(4, max(1, available or 1))


def load_decoder(model_dir="artifacts", *, threads=None, prefer_custom=True, prefer_packed=False):
    """Load the original full-call decoder session."""
    return _load_session(model_dir, threads=threads, prefer_custom=prefer_custom, prefer_packed=prefer_packed)


def load_streaming_decoder(model_dir="artifacts", *, threads=None, prefer_custom=True, prefer_packed=False):
    """Load a prepared streaming graph using the same CPU selection policy."""
    from .streaming import StreamingDecoder
    session, info = _load_session(model_dir, threads=threads, prefer_custom=prefer_custom,
                                  prefer_packed=prefer_packed, streaming=True)
    decoder = StreamingDecoder(session, info.pop("streaming_specification"))
    info["state_bytes_per_stream"] = decoder.state_bytes
    return decoder, info


def _load_session(model_dir, *, threads, prefer_custom, prefer_packed, streaming=False):
    automatic_threads = threads is None
    if automatic_threads:
        threads = _default_threads()
    if isinstance(threads,bool) or not isinstance(threads,int) or threads<1:
        raise ValueError('threads must be a positive integer')
    root=Path(model_dir).resolve()
    manifest=json.loads((root/'bundle.json').read_text())
    machine=platform.machine().lower()
    machine={'aarch64':'arm64','amd64':'x86_64'}.get(machine,machine)
    key=platform.system()+'/'+machine
    selected=manifest['fallback'];library=None;packed_library=None
    info={'platform':key,'selected':'portable_onnx','reason':'standard ONNX fallback',
          'sample_rate':48000,'latent_rate':25,'latent_channels':64,'fresh_call_only':True,
          'threads':threads,'thread_policy':'visible_cpus_capped_at_four' if automatic_threads else 'explicit',
          'onnxruntime':ort.__version__,'tested_runtime':manifest['onnxruntime']}
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    options.add_session_config_entry('session.inter_op.allow_spinning','0')
    native=manifest['native'].get(key) if prefer_custom else None
    required_backend=native.get('required_backend') if native else None
    if required_backend is not None:
        if type(required_backend) is not int or required_backend not in (4,5):
            raise RuntimeError('Invalid required native backend in bundle')
        info['required_backend']=required_backend
        if key!='Linux/x86_64' or native.get('math')!='sleef_u10':
            native=None
            info['reason']='Requested x86 native backend is incompatible with this platform; using standard ONNX'
    if native and ort.__version__!=manifest['onnxruntime']:
        native=None
        info['reason']='Native package requires the tested ORT version; using standard ONNX'
    if native:
        try:library=ctypes.CDLL(str(root/native['library']))
        except OSError as error:
            info['reason']='Native library unavailable on this system; using standard ONNX'
            info['native_load_error']=str(error)
        else:
            library.ncc_abi_version.argtypes=[];library.ncc_abi_version.restype=ctypes.c_uint32
            library.ncc_selected_backend.argtypes=[];library.ncc_selected_backend.restype=ctypes.c_int32
            library.ncc_capabilities.argtypes=[];library.ncc_capabilities.restype=ctypes.c_uint64
            library.ncc_snake_math_name.argtypes=[ctypes.c_int32];library.ncc_snake_math_name.restype=ctypes.c_char_p
            if library.ncc_abi_version()!=1:raise RuntimeError('Native library ABI mismatch')
            backend=required_backend if required_backend is not None else library.ncc_selected_backend()
            caps=library.ncc_capabilities();backend_available=True
            if required_backend is not None:
                try:
                    library.ncc_backend_available.argtypes=[ctypes.c_int32]
                    library.ncc_backend_available.restype=ctypes.c_int32
                    backend_available=bool(library.ncc_backend_available(required_backend))
                except AttributeError:
                    backend_available=False
                if not backend_available:
                    info['reason']='Requested native backend '+str(required_backend)+' is unavailable; using standard ONNX'
            if native['math']=='sleef_u10':
                eligible=False
                if backend_available and bool(caps & 64):
                    try:
                        library.ncc_vector_sine_available.argtypes=[ctypes.c_int32]
                        library.ncc_vector_sine_available.restype=ctypes.c_int32
                        eligible=bool(library.ncc_vector_sine_available(backend))
                    except AttributeError:
                        eligible=False
            elif native['math']=='vforce':eligible=bool(caps & 32) and backend!=1
            else:raise RuntimeError('Unknown native math contract')
            if eligible and native.get('required_math_version') is not None:
                expected = native['required_math_version']
                if type(expected) is not int or expected != 1:
                    raise RuntimeError('Unsupported native streaming math requirement')
                try:
                    library.ncc_streaming_math_version.argtypes = [ctypes.c_int32]
                    library.ncc_streaming_math_version.restype = ctypes.c_uint32
                    info['native_streaming_math_version'] = int(library.ncc_streaming_math_version(backend))
                except AttributeError:
                    info['native_streaming_math_version'] = 0
                eligible = info['native_streaming_math_version'] == expected
            if eligible:
                options.register_custom_ops_library(str(root/native['library']))
                selected=native['model'];info.update(selected='native',reason='Compatible CPU vector-math backend',
                    backend=backend,sine_math=library.ncc_snake_math_name(backend).decode(),
                    experiment=native['experiment'],tested_cpu=native['tested_cpu'])
            elif backend_available:info['reason']='Required vector sine unavailable; using standard ONNX'
    packed=native.get('packed') if native and prefer_packed and info['selected']=='native' else None
    if packed and required_backend==4:
        packed=None
        info['packed_reason']='Explicit AVX2 conflicts with AVX512-only AMD packing; using base native model'
    elif packed and packed.get('required_backend',required_backend)!=required_backend:
        packed=None
        info['packed_reason']='Packed model backend requirement differs from base native model; using base native model'
    if packed:
        if threads not in packed['validated_threads']:
            info['packed_reason']='Thread count not validated for this optional package'
        else:
            try:packed_library=ctypes.CDLL(str(root/packed['library']))
            except OSError as error:
                info['packed_reason']='Optional packed library unavailable; using base native model'
                info['packed_load_error']=str(error)
            else:
                packed_library.ncc_aocl_cpu_supported.argtypes=[]
                packed_library.ncc_aocl_cpu_supported.restype=ctypes.c_int
                packed_library.ncc_aocl_adapter_abi.argtypes=[]
                packed_library.ncc_aocl_adapter_abi.restype=ctypes.c_uint32
                if packed_library.ncc_aocl_adapter_abi()!=1:
                    raise RuntimeError('Packed library ABI mismatch')
                if packed_library.ncc_aocl_cpu_supported():
                    options.register_custom_ops_library(str(root/packed['library']))
                    selected=packed['model']
                    info.update(experiment=packed['experiment'],packed_weights=True,
                                packed_reason='Explicitly requested and compatible CPU',
                                packed_tested_cpu=packed['tested_cpu'])
                else:info['packed_reason']='CPU does not support this optional AMD package'
    if streaming:
        from .assets import sha256
        entry = manifest.get('streaming', {}).get('models', {}).get(selected)
        if not entry:
            raise RuntimeError('No streaming graph for the selected decoder; run fast-audiovae prepare-streaming on this bundle')
        if manifest['streaming'].get('version') != 1:
            raise RuntimeError('Unsupported streaming manifest version')
        if entry.get('required_math_version') is not None:
            if type(entry['required_math_version']) is not int or entry['required_math_version'] != 1:
                raise RuntimeError('Unsupported streaming math requirement')
            if not library or info.get('backend') != entry.get('required_backend', 5):
                raise RuntimeError('Streaming precision requires the validated native backend')
            try:
                library.ncc_streaming_math_version.argtypes = [ctypes.c_int32]
                library.ncc_streaming_math_version.restype = ctypes.c_uint32
                actual = int(library.ncc_streaming_math_version(info['backend']))
            except AttributeError:
                actual = 0
            if actual != entry['required_math_version']:
                raise RuntimeError('Rebuild the native library with consistent streaming sine arithmetic')
        stream_path = (root / entry['model']).resolve()
        if not stream_path.is_relative_to(root) or not stream_path.is_file():
            raise RuntimeError('Streaming graph is missing or outside the model bundle')
        if sha256(root / selected) != entry['source_sha256'] or sha256(stream_path) != entry['model_sha256']:
            raise RuntimeError('Streaming graph or its full-call source differs from the prepared bundle')
        for relative, digest in entry.get('source_external_sha256', {}).items():
            external = (root / relative).resolve()
            if not external.is_relative_to(root) or not external.is_file() or sha256(external) != digest:
                raise RuntimeError('Streaming source external weights differ from the prepared bundle')
        info.update(fresh_call_only=False, streaming=True,
                    streaming_specification=entry, full_call_model=selected)
        selected = entry['model']
    # Explicitly prepared advanced recipes may need several operator libraries.
    # A streaming recipe replaces that list with its state-aware counterparts;
    # registering both versions would collide in the same custom-op domains.
    additional = native.get('additional_libraries', []) if info['selected'] == 'native' else []
    if streaming:
        additional = entry.get('additional_libraries', additional)
    if additional:
        from .assets import sha256
        if not isinstance(additional, list):
            raise RuntimeError('Additional native libraries must be an explicit list')
        registered = set()
        for record in additional:
            path = (root / record['library']).resolve()
            if (not path.is_relative_to(root) or not path.is_file() or path in registered
                    or (native and path == (root / native['library']).resolve())
                    or sha256(path) != record['sha256']):
                raise RuntimeError('Additional native library is missing, duplicated or differs from its manifest')
            registered.add(path)
            options.register_custom_ops_library(str(path))
    session=ort.InferenceSession(str(root/selected),sess_options=options,providers=['CPUExecutionProvider'])
    session.disable_fallback()
    if session.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('CPU-only provider requirement failed')
    session._codec_native_library=library
    session._codec_packed_library=packed_library
    info['providers']=session.get_providers()
    info['model']=selected
    return session,info
