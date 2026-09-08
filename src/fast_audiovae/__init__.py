"""CPU inference for AudioVAE2."""
from .runtime import load_decoder, load_streaming_decoder


def load(*, mode="streaming", threads=1, **options):
    """Automatically prepare and load the CPU decoder; streaming is the default."""
    from .automatic import load as automatic_load
    return automatic_load(mode=mode, threads=threads, **options)


def setup(*, mode="streaming", threads=1, **options):
    """Prepare the CPU cache ahead of the first load, without inference."""
    from .automatic import setup as automatic_setup
    return automatic_setup(mode=mode, threads=threads, **options)


def prepare_encoder(encoder):
    """Apply conservative encoder changes; requires the optional Torch extra."""
    from .encoder import prepare_encoder as prepare
    return prepare(encoder)

__all__ = ["load", "setup", "load_decoder", "load_streaming_decoder", "prepare_encoder"]
