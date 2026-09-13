"""AudioVAE2 inference with CPU defaults and explicit optional Apple GPU use."""
import os as _os

# Avoid starting ORT's telemetry uploader for local CPU decoding. This must
# precede runtime initialization; disabling events later leaves its worker alive.
_os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

from .runtime import load_decoder, load_streaming_decoder


def load(*, mode="streaming", threads=1, device="cpu", **options):
    """Load the decoder. CPU and streaming are defaults; Apple GPU is opt-in."""
    from .automatic import load as automatic_load
    return automatic_load(mode=mode, threads=threads, device=device, **options)


def setup(*, mode="streaming", threads=1, device="cpu", **options):
    """Prepare the selected device ahead of the first load, without inference."""
    from .automatic import setup as automatic_setup
    return automatic_setup(mode=mode, threads=threads, device=device, **options)


def prepare_encoder(encoder):
    """Apply conservative encoder changes; requires the optional Torch extra."""
    from .encoder import prepare_encoder as prepare
    return prepare(encoder)

__all__ = ["load", "setup", "load_decoder", "load_streaming_decoder", "prepare_encoder"]
