import logging
import os
import subprocess
from typing import Optional, List, Dict, Any
import numpy as np

from app.asr_backends.base import ASRBackend, ASRResult
from app.config import get_settings

logger = logging.getLogger(__name__)


class OmnilingualBackend(ASRBackend):
    """
    ASR Backend using facebookresearch/omnilingual-asr.
    """

    def __init__(self):
        self._pipeline = None
        self._model_card = None

    def load(self) -> None:
        """Load the Omnilingual model."""
        if self._pipeline is not None:
            return

        settings = get_settings()
        # Use configured model
        self._model_card = settings.asr_model or "omniASR_CTC_Unlimited_1B_v2"

        logger.info(f"Loading Omnilingual model: {self._model_card}")

        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        # Note: device handling is likely automatic or via env vars in fairseq2
        # but we can check if there are explicit args.
        # The readme doesn't show device args in init, so assuming auto/cuda.
        self._pipeline = ASRInferencePipeline(model_card=self._model_card)

        logger.info("Omnilingual model loaded successfully")

    def decode_audio(self, audio_path: str) -> np.ndarray:
        """
        Decode audio to numpy array (16kHz mono) for the diarization pipeline.
        We use ffmpeg to ensure robustness.
        """
        try:
            # Method 1: Use ffmpeg to read as standard output
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads",
                "0",
                "-i",
                audio_path,
                "-f",
                "s16le",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-",
            ]
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {err.decode()}")

            return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        except Exception as e:
            logger.error(f"Failed to decode audio with ffmpeg: {e}")
            raise

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
        """Transcribe using Omnilingual pipeline."""
        if self._pipeline is None:
            self.load()

        # Language handling:
        # User requested to remove mapping and let model handle it (or use passed language directly).
        # We pass it as a list if provided, otherwise None/empty list?
        # The pipeline.transcribe signature expects lang=[...].

        lang_input = [language] if language else None

        logger.info(f"Transcribing {audio_path} with language input: {lang_input}")

        # Usage: transcriptions = pipeline.transcribe(audio_files, lang=lang, batch_size=2)
        transcriptions = self._pipeline.transcribe(
            [audio_path], lang=lang_input, batch_size=1
        )

        # Result is likely a list of strings (transcripts)
        text = transcriptions[0]

        # We need duration.
        audio_data = self.decode_audio(audio_path)
        duration = len(audio_data) / 16000.0

        return ASRResult(
            text=text,
            language=language or "unknown",
            duration=duration,
            segments=[{"start": 0.0, "end": duration, "text": text}],
            words=[],  # No word timestamps from generic pipeline
        )

    def unload(self):
        if self._pipeline:
            del self._pipeline
            self._pipeline = None
