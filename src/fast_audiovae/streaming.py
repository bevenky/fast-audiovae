"""Independent, bounded decoder state for the explicit-state ONNX interface."""
from __future__ import annotations

import threading

import numpy as np


class StreamingDecoder:
    """Share immutable weights and one CPU session across independent streams.

    Use ``with decoder.streaming_decode() as stream`` and pass only new latent
    frames to ``stream.decode_chunk``. A latent frame produces 1920 samples.
    """

    def __init__(self, session, specification):
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("Streaming requires a CPU-only ONNX session")
        self._session = session
        self._latent = specification["latent_input"]
        self._audio = specification["audio_output"]
        states = specification["states"]
        if not isinstance(states, list) or not states:
            raise ValueError("Streaming graph must declare explicit history states")
        self._states = []
        for item in states:
            shape = item["shape"]
            if (item.get("dtype") != "float32" or not isinstance(shape, list)
                    or len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape)
                    or shape[0] != 1):
                raise ValueError("Invalid streaming state shape or type")
            self._states.append((item["input"], item["output"], tuple(shape)))
        inputs = {i.name: i for i in session.get_inputs()}
        outputs = {i.name: i for i in session.get_outputs()}
        input_names = [self._latent] + [i for i, _, _ in self._states]
        output_names = [self._audio] + [o for _, o, _ in self._states]
        if (len(set(input_names)) != len(input_names) or len(set(output_names)) != len(output_names)
                or set(inputs) != set(input_names) or set(outputs) != set(output_names)):
            raise ValueError("Streaming manifest and graph inputs/outputs differ")
        for name, items, channels in ((self._latent, inputs, 64), (self._audio, outputs, 1)):
            value = items[name]
            if value.type != "tensor(float)" or len(value.shape) != 3 or value.shape[:2] != [1, channels]:
                raise ValueError("Unexpected streaming latent/audio contract")
        for inp, out, shape in self._states:
            for name, items in ((inp, inputs), (out, outputs)):
                if items[name].type != "tensor(float)" or tuple(items[name].shape) != shape:
                    raise ValueError("Streaming history does not match graph: " + name)
        self._outputs = output_names
        self.state_bytes = sum(int(np.prod(s)) * 4 for _, _, s in self._states)

    def streaming_decode(self):
        """Create a fresh stream. Streams never share mutable audio history."""
        return DecoderStream(self)


class DecoderStream:
    """A single ordered sequence of latent chunks, with bounded memory."""

    def __init__(self, decoder):
        self._decoder = decoder
        self._lock = threading.Lock()
        self._closed = False
        self._history = self._zeros()
        self.frames_decoded = 0

    def _zeros(self):
        return {i: np.zeros(shape, dtype=np.float32) for i, _, shape in self._decoder._states}

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Decoder stream is closed; create a new stream")

    def decode_chunk(self, latents):
        """Decode FP32 [1,64,L] new frames; never resend previous frames.

        Failed validation or inference leaves the previous state intact.
        Empty chunks return empty audio without advancing the stream.
        """
        if not isinstance(latents, np.ndarray) or latents.dtype != np.float32:
            raise ValueError("Latents must be a float32 NumPy array")
        if latents.ndim != 3 or latents.shape[:2] != (1, 64):
            raise ValueError("Latents must have shape [1,64,L]")
        if not np.isfinite(latents).all():
            raise ValueError("Latents must contain only finite values")
        with self._lock:
            self._check_open()
            frames = latents.shape[2]
            if not frames:
                return np.empty((1, 1, 0), dtype=np.float32)
            decoder = self._decoder
            feed = {decoder._latent: np.ascontiguousarray(latents), **self._history}
            values = decoder._session.run(decoder._outputs, feed)
            if len(values) != len(decoder._outputs):
                raise RuntimeError("Streaming graph returned an incomplete result")
            audio = values[0]
            if (audio.dtype != np.float32 or audio.shape != (1, 1, frames * 1920)
                    or not np.isfinite(audio).all()):
                raise RuntimeError("Streaming graph returned invalid audio")
            history = {}
            for (inp, _, shape), value in zip(decoder._states, values[1:]):
                if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                    raise RuntimeError("Streaming graph returned invalid history: " + inp)
                history[inp] = value
            self._history = history
            self.frames_decoded += frames
            return audio

    def reset(self):
        """Start a new utterance without rebuilding the session or its weights."""
        with self._lock:
            self._check_open()
            self._history = self._zeros()
            self.frames_decoded = 0

    def flush(self):
        """Return no extra tail: all samples are emitted with their latent frame.

        AudioVAE2's causal decoder has no delayed output after a complete latent
        frame. This does not close or reset the stream.
        """
        with self._lock:
            self._check_open()
            return np.empty((1, 1, 0), dtype=np.float32)

    def close(self):
        with self._lock:
            self._history.clear()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()
