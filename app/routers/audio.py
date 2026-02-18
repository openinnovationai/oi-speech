"""
Audio transcription router with speaker diarization.
"""

import logging
import tempfile
import os
import asyncio
from typing import Optional, Literal

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse

from app.models.audio import (
    TranscriptionResponse,
    VerboseTranscriptionResponse,
    DiarizedTranscriptionResponse,
    DiarizedSegment,
    DiarizedWord,
    Segment,
    DiarizationResponse,
    SpeakerSegment,
)
from app.services.transcription import (
    transcribe_with_diarization,
    transcribe_audio,
    get_device_and_compute_type,
)
from app.services.diarization import get_diarizer
from app.services.helpers import write_srt, write_vtt, get_speaker_aware_transcript
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["audio"])

ResponseFormat = Literal["json", "text", "srt", "vtt", "verbose_json"]


def _run_diarized_transcription(
    audio_path: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    task: str,
):
    """Run diarized transcription synchronously (to be called from a thread)."""
    return transcribe_with_diarization(
        audio_path=audio_path,
        language=language,
        prompt=prompt,
        temperature=temperature,
        task=task,
    )


def _run_simple_transcription(
    audio_path: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    word_timestamps: bool,
    task: str,
):
    """Run simple transcription synchronously (to be called from a thread)."""
    segments_gen, info = transcribe_audio(
        audio_path=audio_path,
        language=language,
        prompt=prompt,
        temperature=temperature,
        word_timestamps=word_timestamps,
        task=task,
    )
    segments_list = list(segments_gen)
    return segments_list, info


@router.post("/transcriptions")
async def create_transcription(
    file: UploadFile = File(..., description="The audio file to transcribe"),
    model: str = Form(
        ..., description="Model to use (accepted for compatibility, uses server model)"
    ),
    language: Optional[str] = Form(
        None, description="Language of the audio (ISO-639-1 code)"
    ),
    prompt: Optional[str] = Form(
        None, description="Optional prompt to guide transcription"
    ),
    response_format: ResponseFormat = Form("json", description="Output format"),
    temperature: float = Form(0.0, description="Sampling temperature (0-1)"),
    diarize: bool = Form(True, description="Enable speaker diarization"),
):
    """
    Transcribe audio into text with speaker diarization.

    OpenAI-compatible endpoint for audio transcription with added diarization support.
    """
    temp_file = None
    try:
        # Create temp file with proper extension
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

        # Write uploaded content
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        logger.info(
            f"Processing audio file: {file.filename}, format: {response_format}, diarize: {diarize}"
        )

        if diarize:
            # Run diarized transcription
            result = await asyncio.to_thread(
                _run_diarized_transcription,
                temp_file.name,
                language,
                prompt,
                temperature,
                "transcribe",
            )

            # Return based on format
            if response_format == "text":
                return PlainTextResponse(content=result["text"])

            elif response_format == "srt":
                srt_content = write_srt(result["segments"])
                return PlainTextResponse(content=srt_content, media_type="text/plain")

            elif response_format == "vtt":
                vtt_content = write_vtt(result["segments"])
                return PlainTextResponse(content=vtt_content, media_type="text/vtt")

            elif response_format == "verbose_json":
                return DiarizedTranscriptionResponse(
                    task="transcribe",
                    language=result["language"],
                    duration=result["duration"],
                    text=result["text"],
                    segments=[
                        DiarizedSegment(
                            speaker=seg["speaker"],
                            start_time=seg["start_time"],
                            end_time=seg["end_time"],
                            text=seg["text"].strip(),
                        )
                        for seg in result["segments"]
                    ],
                    words=[
                        DiarizedWord(
                            word=w["word"],
                            start_time=w["start_time"],
                            end_time=w["end_time"],
                            speaker=w["speaker"],
                        )
                        for w in result["words"]
                    ],
                )

            else:  # json (default)
                return TranscriptionResponse(text=result["text"])

        else:
            # Run simple transcription without diarization
            segments_list, info = await asyncio.to_thread(
                _run_simple_transcription,
                temp_file.name,
                language,
                prompt,
                temperature,
                response_format == "verbose_json",
                "transcribe",
            )

            full_text = " ".join(seg.text.strip() for seg in segments_list)

            if response_format == "text":
                return PlainTextResponse(content=full_text)

            elif response_format == "srt":
                srt_lines = []
                for i, seg in enumerate(segments_list, 1):
                    start = _format_srt_time(seg.start)
                    end = _format_srt_time(seg.end)
                    srt_lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
                return PlainTextResponse(
                    content="\n".join(srt_lines), media_type="text/plain"
                )

            elif response_format == "vtt":
                vtt_lines = ["WEBVTT\n"]
                for seg in segments_list:
                    start = _format_vtt_time(seg.start)
                    end = _format_vtt_time(seg.end)
                    vtt_lines.append(f"{start} --> {end}\n{seg.text.strip()}\n")
                return PlainTextResponse(
                    content="\n".join(vtt_lines), media_type="text/vtt"
                )

            elif response_format == "verbose_json":
                return VerboseTranscriptionResponse(
                    task="transcribe",
                    language=info.language,
                    duration=info.duration,
                    text=full_text,
                    segments=[
                        Segment(
                            id=i,
                            seek=int(seg.seek),
                            start=seg.start,
                            end=seg.end,
                            text=seg.text,
                            tokens=list(seg.tokens) if seg.tokens else [],
                            temperature=seg.temperature,
                            avg_logprob=seg.avg_logprob,
                            compression_ratio=seg.compression_ratio,
                            no_speech_prob=seg.no_speech_prob,
                        )
                        for i, seg in enumerate(segments_list)
                    ],
                )

            else:  # json (default)
                return TranscriptionResponse(text=full_text)

    except Exception as e:
        # Handle empty transcript gracefully - return 200 with empty response
        if "Empty transcript" in str(e):
            logger.warning(f"Empty transcript received, returning empty response")
            return TranscriptionResponse(text="")
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@router.post("/translations")
async def create_translation(
    file: UploadFile = File(..., description="The audio file to translate"),
    model: str = Form(
        ..., description="Model to use (accepted for compatibility, uses server model)"
    ),
    prompt: Optional[str] = Form(
        None, description="Optional prompt to guide translation"
    ),
    response_format: ResponseFormat = Form("json", description="Output format"),
    temperature: float = Form(0.0, description="Sampling temperature (0-1)"),
    diarize: bool = Form(True, description="Enable speaker diarization"),
):
    """
    Translate audio into English text with speaker diarization.

    OpenAI-compatible endpoint for audio translation with added diarization support.
    """
    temp_file = None
    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        logger.info(
            f"Translating audio file: {file.filename}, format: {response_format}, diarize: {diarize}"
        )

        if diarize:
            result = await asyncio.to_thread(
                _run_diarized_transcription,
                temp_file.name,
                None,  # Auto-detect source language
                prompt,
                temperature,
                "translate",
            )

            if response_format == "text":
                return PlainTextResponse(content=result["text"])

            elif response_format == "srt":
                srt_content = write_srt(result["segments"])
                return PlainTextResponse(content=srt_content, media_type="text/plain")

            elif response_format == "vtt":
                vtt_content = write_vtt(result["segments"])
                return PlainTextResponse(content=vtt_content, media_type="text/vtt")

            elif response_format == "verbose_json":
                return DiarizedTranscriptionResponse(
                    task="translate",
                    language=result["language"],
                    duration=result["duration"],
                    text=result["text"],
                    segments=[
                        DiarizedSegment(
                            speaker=seg["speaker"],
                            start_time=seg["start_time"],
                            end_time=seg["end_time"],
                            text=seg["text"].strip(),
                        )
                        for seg in result["segments"]
                    ],
                    words=[
                        DiarizedWord(
                            word=w["word"],
                            start_time=w["start_time"],
                            end_time=w["end_time"],
                            speaker=w["speaker"],
                        )
                        for w in result["words"]
                    ],
                )

            else:
                return TranscriptionResponse(text=result["text"])

        else:
            segments_list, info = await asyncio.to_thread(
                _run_simple_transcription,
                temp_file.name,
                None,
                prompt,
                temperature,
                response_format == "verbose_json",
                "translate",
            )

            full_text = " ".join(seg.text.strip() for seg in segments_list)

            if response_format == "text":
                return PlainTextResponse(content=full_text)
            elif response_format == "verbose_json":
                return VerboseTranscriptionResponse(
                    task="translate",
                    language=info.language,
                    duration=info.duration,
                    text=full_text,
                    segments=[
                        Segment(
                            id=i,
                            seek=int(seg.seek),
                            start=seg.start,
                            end=seg.end,
                            text=seg.text,
                            tokens=list(seg.tokens) if seg.tokens else [],
                            temperature=seg.temperature,
                            avg_logprob=seg.avg_logprob,
                            compression_ratio=seg.compression_ratio,
                            no_speech_prob=seg.no_speech_prob,
                        )
                        for i, seg in enumerate(segments_list)
                    ],
                )
            else:
                return TranscriptionResponse(text=full_text)

    except Exception as e:
        # Handle empty transcript gracefully - return 200 with empty response
        if "Empty transcript" in str(e):
            logger.warning(f"Empty transcript received, returning empty response")
            return TranscriptionResponse(text="")
        logger.error(f"Translation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format seconds to VTT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _run_diarization_only(audio_path: str):
    """Run diarization only (no transcription) synchronously."""
    import torch
    from app.asr_backends.factory import get_backend

    device, _ = get_device_and_compute_type()
    diarizer = get_diarizer(device)

    # Load audio waveform
    backend = get_backend()
    audio_waveform = backend.decode_audio(audio_path)
    duration = len(audio_waveform) / 16000  # 16kHz sample rate

    # Run diarization
    speaker_ts = diarizer.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))

    return speaker_ts, duration


@router.post("/diarizations")
async def create_diarization(
    file: UploadFile = File(..., description="The audio file to diarize"),
    model: str = Form(
        "whisper-1", description="Model to use (accepted for compatibility)"
    ),
):
    """
    Perform speaker diarization on audio without transcription.

    Returns speaker segments with timestamps indicating when each speaker is talking.
    This is useful when you only need to identify who spoke when, without the text.
    """
    temp_file = None
    try:
        # Create temp file with proper extension
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

        # Write uploaded content
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        logger.info(f"Processing audio file for diarization only: {file.filename}")

        # Run diarization in thread pool
        speaker_ts, duration = await asyncio.to_thread(
            _run_diarization_only,
            temp_file.name,
        )

        # Extract unique speakers
        speakers = list(set(seg[2] for seg in speaker_ts))
        speakers.sort()  # Sort for consistent ordering

        # Build response
        segments = [
            SpeakerSegment(
                speaker=seg[2],
                start_time=seg[0],  # Already in milliseconds from diarizer
                end_time=seg[1],
            )
            for seg in speaker_ts
        ]

        return DiarizationResponse(
            duration=duration,
            speakers=speakers,
            segments=segments,
        )

    except Exception as e:
        logger.error(f"Diarization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")

    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
