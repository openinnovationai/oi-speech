"""
Transcription service with diarization support.
"""

import logging
import os
import tempfile
from typing import Optional, Tuple, List, Generator, Dict, Any

import torch

from app.config import get_settings
from app.services.helpers import (
    get_words_speaker_mapping,
    get_sentences_speaker_mapping,
    get_realigned_ws_mapping_with_punctuation,
    get_speaker_aware_transcript,
    langs_to_iso,
    punct_model_langs,
)
from app.services.diarization import get_diarizer
from app.asr_backends.factory import get_backend, load_backend as load_asr_backend

logger = logging.getLogger(__name__)

# Global model instances (for post-processing)
_alignment_model = None
_alignment_tokenizer = None
_punct_model = None


def get_device_and_compute_type() -> Tuple[str, str]:
    """Determine the device and compute type based on settings and availability."""
    settings = get_settings()
    device = settings.asr_device
    compute_type = settings.asr_compute_type

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type == "auto":
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"

    return device, compute_type


def load_alignment_model():
    """Load the CTC forced alignment model."""
    global _alignment_model, _alignment_tokenizer

    if _alignment_model is not None:
        return _alignment_model, _alignment_tokenizer

    from ctc_forced_aligner import load_alignment_model as load_ctc_model

    device, _ = get_device_and_compute_type()
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"Loading alignment model on {device}")
    _alignment_model, _alignment_tokenizer = load_ctc_model(device, dtype=dtype)
    logger.info("Alignment model loaded successfully")

    return _alignment_model, _alignment_tokenizer


def load_punct_model():
    """Load the punctuation restoration model."""
    global _punct_model

    if _punct_model is not None:
        return _punct_model

    from deepmultilingualpunctuation import PunctuationModel

    logger.info("Loading punctuation model")
    _punct_model = PunctuationModel(model="kredor/punctuate-all")
    logger.info("Punctuation model loaded successfully")

    return _punct_model


def load_models():
    """Load all required models."""
    settings = get_settings()
    device, _ = get_device_and_compute_type()

    # Load ASR Backend
    load_asr_backend()

    # Load support models
    load_alignment_model()
    load_punct_model()
    get_diarizer(device)


def transcribe_with_diarization(
    audio_path: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    task: str = "transcribe",
) -> Dict[str, Any]:
    """
    Transcribe audio with speaker diarization.

    Args:
        audio_path: Path to the audio file
        language: Optional language code (ISO-639-1)
        prompt: Optional prompt to guide transcription
        temperature: Sampling temperature
        task: "transcribe" or "translate"

    Returns:
        Dictionary containing:
        - text: Speaker-aware transcript
        - language: Detected language
        - duration: Audio duration
        - segments: List of diarized sentence segments
        - words: List of word-speaker mappings
    """
    import re

    settings = get_settings()
    device, compute_type = get_device_and_compute_type()

    # Get ASR Backend
    backend = get_backend()

    # Transcribe (Step 1)
    # Note: Backend handles language processing and decoding
    result = backend.transcribe(
        audio_path,
        language=language,
        prompt=prompt,
        temperature=temperature,
        task=task,
        word_timestamps=False,  # We use forced alignment for precision unless backend is super good
    )

    full_transcript = result.text
    logger.info(f"Transcription complete. Language: {result.language}")

    # For alignment and diarization, we need the waveform.
    # We ask the backend to decode it for us to ensure consistency.
    audio_waveform = backend.decode_audio(audio_path)

    # Forced alignment for word timestamps (Step 2)
    logger.info("Running forced alignment...")
    alignment_model, alignment_tokenizer = load_alignment_model()

    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        postprocess_results,
        preprocess_text,
    )

    emissions, stride = generate_emissions(
        alignment_model,
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device),
        batch_size=settings.batch_size if settings.batch_size > 0 else 8,
    )

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso.get(result.language, "eng"),
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    logger.info(f"Alignment complete. Found {len(word_timestamps)} word timestamps")

    # Speaker diarization (Step 3)
    logger.info("Running speaker diarization...")
    diarizer = get_diarizer(device)
    speaker_ts = diarizer.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
    logger.info(f"Diarization complete. Found {len(speaker_ts)} speaker segments")

    # Map words to speakers
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    # Punctuation restoration (Step 4)
    if result.language in punct_model_langs:
        logger.info("Restoring punctuation...")
        punct_model = load_punct_model()
        words_list = [x["word"] for x in wsm]
        labeled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    # Realign based on punctuation
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)

    # Get sentence-level speaker mapping
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Generate speaker-aware transcript
    text = get_speaker_aware_transcript(ssm)

    return {
        "text": text,
        "language": result.language,
        "duration": result.duration,
        "segments": ssm,
        "words": wsm,
    }


def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    word_timestamps: bool = False,
    task: str = "transcribe",
) -> Tuple[List[Any], Any]:
    """
    Simple transcription without diarization (for compatibility).

    Returns:
        Tuple of (list of segments, transcription info object)
    """
    backend = get_backend()

    # We map word_timestamps arg.
    # For ASRResult, we need to convert it back to something compatible
    # with the router's expectations (which expects object with .start, .end, .text)

    result = backend.transcribe(
        audio_path,
        language=language,
        prompt=prompt,
        temperature=temperature,
        word_timestamps=word_timestamps,
        task=task,
    )

    # Create valid response objects compatible with existing router logic
    # The router expects an object 'info' with .language and .duration
    class Info:
        def __init__(self, language, duration):
            self.language = language
            self.duration = duration

    info = Info(result.language, result.duration)

    # The router expects segments to have .start, .end, .text attributes
    # ASRResult.segments is a list of dicts. We wrap them.
    class SegmentObj:
        def __init__(self, data):
            self.start = data["start"]
            self.end = data["end"]
            self.text = data["text"]
            self.words = []  # TODO if needed

            # Add dummy attributes that might be accessed by verbose_json response
            self.id = 0
            self.seek = 0
            self.tokens = []
            self.temperature = 0.0
            self.avg_logprob = 0.0
            self.compression_ratio = 0.0
            self.no_speech_prob = 0.0

    segments = [SegmentObj(s) for s in result.segments]

    return segments, info
