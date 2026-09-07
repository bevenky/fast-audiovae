"""Pinned model downloads. Nothing is downloaded when importing the package."""
import hashlib
from pathlib import Path
import urllib.request

MODEL_REVISION = "ecb511b96675f041424b42f148bf72e301262586"
MODEL_BASE = "https://huggingface.co/ai4all8/VoxCPM2-ONNX/resolve/" + MODEL_REVISION + "/"
MODEL_FILES = {
    "audio_vae_decoder.onnx": "fdd20e200675ab9649bf33f16bc5974101ba5cf47ab89fbfe29f3d36ff6c3c84",
    "audio_vae_decoder.onnx.data": "3b76bd57356e96893655bd24a364989e016c5af2fc8c29195caaefb4eccf5143",
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(url, destination, expected):
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError("Existing file has a different checksum: " + str(destination))
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
        if sha256(temporary) != expected:
            raise ValueError("Downloaded file has an unexpected checksum: " + url)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def fetch_model(directory):
    directory = Path(directory)
    for name, digest in MODEL_FILES.items():
        download(MODEL_BASE + name, directory / name, digest)
    return directory / "audio_vae_decoder.onnx"


def verify_model(source):
    source = Path(source)
    for path, expected in ((source, MODEL_FILES["audio_vae_decoder.onnx"]),
                           (source.parent / "audio_vae_decoder.onnx.data", MODEL_FILES["audio_vae_decoder.onnx.data"])):
        if sha256(path) != expected:
            raise ValueError("This preparation recipe requires the pinned AudioVAE2 export: " + str(path))
    return source
