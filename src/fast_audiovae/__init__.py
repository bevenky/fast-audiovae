"""CPU inference for AudioVAE2."""
from .runtime import load_decoder, load_streaming_decoder


def prepare_encoder(encoder):
    """Apply conservative encoder changes; requires the optional Torch extra."""
    from .encoder import prepare_encoder as prepare
    return prepare(encoder)

__all__ = ["load_decoder", "load_streaming_decoder", "prepare_encoder"]
