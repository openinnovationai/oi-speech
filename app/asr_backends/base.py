from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import numpy as np

@dataclass
class ASRResult:
    """Standardized output from any ASR backend."""
    text: str
    language: str
    duration: float
    segments: List[Dict[str, Any]]  # List of {start, end, text, ...} dicts
    words: Optional[List[Dict[str, Any]]] = None  # List of {word, start, end, ...} dicts

class ASRBackend(ABC):
    """Abstract base class for ASR backends."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None, 
        prompt: Optional[str] = None, 
        temperature: float = 0.0,
        word_timestamps: bool = False,
        task: str = "transcribe",
        **kwargs
    ) -> ASRResult:
        """
        Transcribe an audio file and return standardized results.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (ISO-639-1)
            prompt: Optional prompt to guide transcription
            temperature: Sampling temperature
            word_timestamps: Whether to return word-level timestamps
            task: "transcribe" or "translate"
            kwargs: Backend-specific arguments
            
        Returns:
            ASRResult object containing text, segments, etc.
        """
        pass

    @abstractmethod
    def decode_audio(self, audio_path: str) -> np.ndarray:
        """
        Decode audio file to numpy waveform (16kHz mono).
        Used by the diarization pipeline.
        """
        pass

    def unload(self) -> None:
        """Optional: free GPU memory."""
        pass
