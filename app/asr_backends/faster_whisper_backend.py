import logging
import os
from typing import Optional, List, Any, Dict
import numpy as np
import torch
import faster_whisper

from app.config import get_settings
from app.asr_backends.base import ASRBackend, ASRResult
from app.services.helpers import process_language_arg

logger = logging.getLogger(__name__)


class FasterWhisperBackend(ASRBackend):
    """
    ASR Backend using faster-whisper.
    """

    def __init__(self):
        self._model: Optional[faster_whisper.WhisperModel] = None

    def load(self) -> None:
        """Load the Whisper model based on configuration."""
        if self._model is not None:
            return

        settings = get_settings()

        # Determine device and compute type
        device = settings.asr_device
        compute_type = settings.asr_compute_type

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if compute_type == "auto":
            if device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"

        # Use ASR_MODEL directly
        model_name = settings.asr_model

        logger.info(
            f"Loading Faster Whisper model: {model_name} on {device} with {compute_type}"
        )

        self._model = faster_whisper.WhisperModel(
            model_name, device=device, compute_type=compute_type
        )

        logger.info("Faster Whisper model loaded successfully")

    def decode_audio(self, audio_path: str) -> np.ndarray:
        """Decode audio to numpy array using faster_whisper utility."""
        return faster_whisper.decode_audio(audio_path)

    def _find_numeral_symbol_tokens(self) -> List[int]:
        """Find token IDs that contain numerals or currency symbols."""
        if not self._model:
            return [-1]

        tokenizer = self._model.hf_tokenizer
        numeral_symbol_tokens = [-1]
        for token, token_id in tokenizer.get_vocab().items():
            has_numeral_symbol = any(c in "0123456789%$£" for c in token)
            if has_numeral_symbol:
                numeral_symbol_tokens.append(token_id)
        return numeral_symbol_tokens

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        word_timestamps: bool = False,
        task: str = "transcribe",
        **kwargs,
    ) -> ASRResult:
        """Transcribe using faster-whisper."""
        if self._model is None:
            self.load()

        settings = get_settings()

        # Process language arg
        model_name = settings.asr_model
        language = process_language_arg(language, model_name)

        # Load audio
        audio_waveform = self.decode_audio(audio_path)
        duration = len(audio_waveform) / 16000

        # Create pipeline
        # Note: We recreate pipeline per request or could cache it?
        # Original code recreated it. Ideally we cache it but BatchedInferencePipeline
        # wraps the model.
        whisper_pipeline = faster_whisper.BatchedInferencePipeline(self._model)

        # Suppress numerals
        suppress_tokens = [-1]
        if settings.suppress_numerals:
            suppress_tokens = self._find_numeral_symbol_tokens()

        logger.info(
            f"Starting specific transcription with faster-whisper (task={task})..."
        )

        # Run transcription
        if settings.batch_size > 0:
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                batch_size=settings.batch_size,
                initial_prompt=prompt,
                temperature=temperature,
                task=task,
                word_timestamps=word_timestamps,
            )
        else:
            # Fallback to non-batched
            transcript_segments, info = self._model.transcribe(
                audio_waveform,
                language=language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
                initial_prompt=prompt,
                temperature=temperature,
                task=task,
                word_timestamps=word_timestamps,
            )

        # Convert segments to list and standardized format
        segments_list = list(transcript_segments)
        full_text = "".join(segment.text for segment in segments_list)

        # Simplify segments for ASRResult
        simple_segments = []
        all_words = []

        for seg in segments_list:
            simple_segments.append(
                {"start": seg.start, "end": seg.end, "text": seg.text}
            )
            if word_timestamps and seg.words:
                for w in seg.words:
                    all_words.append(
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                    )

        return ASRResult(
            text=full_text,
            language=info.language,
            duration=info.duration,  # or calculated duration
            segments=simple_segments,
            words=all_words if word_timestamps else None,
        )

    def unload(self):
        if self._model:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
