from pydantic import BaseModel
from typing import List, Optional


class Word(BaseModel):
    """A word with timing and speaker information."""

    word: str
    start: float
    end: float
    probability: Optional[float] = None


class DiarizedWord(BaseModel):
    """A word with timing and speaker information."""

    word: str
    start_time: int  # milliseconds
    end_time: int  # milliseconds
    speaker: str


class Segment(BaseModel):
    """A transcription segment with timing information."""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    speaker: Optional[str] = None


class DiarizedSegment(BaseModel):
    """A sentence segment with speaker and timing information."""

    speaker: str
    start_time: float  # seconds
    end_time: float  # seconds
    text: str


class TranscriptionResponse(BaseModel):
    """Simple transcription response (default format)."""

    text: str


class VerboseTranscriptionResponse(BaseModel):
    """Verbose transcription response with segments and speaker information."""

    task: str
    language: str
    duration: float
    text: str
    segments: List[Segment]


class DiarizedTranscriptionResponse(BaseModel):
    """Full diarized transcription response with speaker-aware sentences."""

    task: str
    language: str
    duration: float
    text: str
    segments: List[DiarizedSegment]
    words: Optional[List[DiarizedWord]] = None


class SpeakerSegment(BaseModel):
    """A speaker segment with timing information."""

    speaker: str
    start_time: float  # milliseconds
    end_time: float  # milliseconds


class DiarizationResponse(BaseModel):
    """Response for diarization-only endpoint."""

    duration: float
    speakers: List[str]
    segments: List[SpeakerSegment]
