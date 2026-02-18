from functools import lru_cache
import os
from typing import Optional

from app.config import get_settings
from app.asr_backends.base import ASRBackend
from app.asr_backends.faster_whisper_backend import FasterWhisperBackend
from app.asr_backends.omnilingual_backend import OmnilingualBackend


@lru_cache()
def get_backend() -> ASRBackend:
    """
    Get the configured ASR backend instance.
    Cached to ensure singleton behavior across requests.
    """
    settings = get_settings()
    backend_type = os.getenv("ASR_BACKEND", "faster_whisper").lower()

    if backend_type == "faster_whisper":
        return FasterWhisperBackend()
    elif backend_type == "omnilingual":
        return OmnilingualBackend()
    else:
        raise ValueError(f"Unsupported ASR backend: {backend_type}")


def load_backend():
    """Pre-load the backend model."""
    backend = get_backend()
    backend.load()
