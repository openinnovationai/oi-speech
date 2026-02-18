from pydantic_settings import BaseSettings
from typing import Literal, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ASR Configuration
    asr_backend: Literal["faster_whisper", "omnilingual"] = "faster_whisper"
    asr_model: str = "medium.en"
    asr_device: Literal["auto", "cpu", "cuda"] = "auto"
    asr_compute_type: Literal[
        "auto", "int8", "int8_float16", "int8_float32", "float16", "float32"
    ] = "auto"

    # Diarization settings
    enable_stemming: bool = False  # Use Demucs for vocal separation
    suppress_numerals: bool = True  # Transcribe numbers as words
    batch_size: int = 8  # Batch size for batched inference

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_prefix = ""
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
