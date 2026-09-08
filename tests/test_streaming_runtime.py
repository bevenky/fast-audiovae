"""Stream lifecycle, isolation and transactional state advancement."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from fast_audiovae.streaming import StreamingDecoder


SPEC = {"latent_input": "z", "audio_output": "audio", "states": [
    {"input": "history", "output": "next_history", "shape": [1, 1, 1], "dtype": "float32"}]}


class Session:
    """One causal delay followed by hold upsampling, with no internal state."""
    fail = False

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name="z", shape=[1, 64, "L"], type="tensor(float)"),
                SimpleNamespace(name="history", shape=[1, 1, 1], type="tensor(float)")]

    def get_outputs(self):
        return [SimpleNamespace(name="audio", shape=[1, 1, "samples"], type="tensor(float)"),
                SimpleNamespace(name="next_history", shape=[1, 1, 1], type="tensor(float)")]

    def run(self, outputs, feed):
        if self.fail:
            raise RuntimeError("inference failed")
        x = feed["z"][:, :1]
        previous = np.concatenate([feed["history"], x[:, :, :-1]], axis=2)
        return [np.repeat(x + previous, 1920, axis=2), x[:, :, -1:].copy()]


def z(*values):
    return np.tile(np.array(values, dtype=np.float32)[None, None, :], (1, 64, 1))


def test_chunked_reset_and_independent_streams_match_full():
    decoder = StreamingDecoder(Session(), SPEC)
    with decoder.streaming_decode() as full:
        expected = full.decode_chunk(z(1, 2, 3, 4, 5))
    with decoder.streaming_decode() as first, decoder.streaming_decode() as second:
        a = first.decode_chunk(z(1, 2))
        b = second.decode_chunk(z(8, 9))
        c = first.decode_chunk(z(3, 4, 5))
        np.testing.assert_array_equal(np.concatenate([a, c], axis=2), expected)
        assert b[0, 0, 0] == 8
        assert first.frames_decoded == 5
        first.reset()
        np.testing.assert_array_equal(first.decode_chunk(z(1, 2, 3, 4, 5)), expected)
        assert decoder.state_bytes == 4
        assert first.flush().shape == (1, 1, 0)


def test_empty_noncontiguous_and_closed_stream():
    decoder = StreamingDecoder(Session(), SPEC)
    stream = decoder.streaming_decode()
    assert stream.decode_chunk(z()).shape == (1, 1, 0)
    assert stream.frames_decoded == 0
    np.testing.assert_array_equal(stream.decode_chunk(z(1, 9, 2)[:, :, ::2]),
                                  np.repeat(np.array([[[1, 3]]], dtype=np.float32), 1920, axis=2))
    stream.close()
    stream.close()
    for action in (lambda: stream.decode_chunk(z(1)), stream.reset, stream.flush):
        with pytest.raises(RuntimeError, match="closed"):
            action()


def test_invalid_input_and_failed_inference_do_not_advance():
    session = Session()
    stream = StreamingDecoder(session, SPEC).streaming_decode()
    stream.decode_chunk(z(7))
    for bad in (z(1).astype(np.float64), z(float("nan")), np.ones((1, 63, 1), np.float32), [1]):
        with pytest.raises(ValueError):
            stream.decode_chunk(bad)
        assert stream.frames_decoded == 1
    session.fail = True
    with pytest.raises(RuntimeError, match="inference failed"):
        stream.decode_chunk(z(8))
    session.fail = False
    assert stream.decode_chunk(z(2))[0, 0, 0] == 9


def test_parallel_streams_share_session_but_not_history():
    decoder = StreamingDecoder(Session(), SPEC)
    def run(value):
        with decoder.streaming_decode() as stream:
            stream.decode_chunk(z(value))
            return stream.decode_chunk(z(value + 1))[0, 0, 0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run, range(16))) == [2 * i + 1 for i in range(16)]


def test_wrong_manifest_or_provider_is_rejected():
    session = Session()
    wrong = {**SPEC, "latent_input": "other"}
    with pytest.raises(ValueError, match="inputs/outputs"):
        StreamingDecoder(session, wrong)
    session.get_providers = lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"]
    with pytest.raises(ValueError, match="CPU-only"):
        StreamingDecoder(session, SPEC)


def test_invalid_output_does_not_publish_any_new_state():
    session = Session()
    stream = StreamingDecoder(session, SPEC).streaming_decode()
    stream.decode_chunk(z(4))
    run = session.run
    session.run = lambda *_: [np.zeros((1, 1, 1920), np.float32), np.full((1, 1, 1), np.nan, np.float32)]
    with pytest.raises(RuntimeError, match="invalid history"):
        stream.decode_chunk(z(10))
    session.run = run
    assert stream.decode_chunk(z(2))[0, 0, 0] == 6
