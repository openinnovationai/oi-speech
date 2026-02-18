"""
Whisper Diarization API - FastAPI application with speaker diarization.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import audio
from app.services.transcription import load_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: Load all models
    logger.info("Starting Whisper Diarization API server...")
    settings = get_settings()
    logger.info(
        f"Configuration: backend={settings.asr_backend}, model={settings.asr_model}, "
        f"device={settings.asr_device}, compute_type={settings.asr_compute_type}, stemming={settings.enable_stemming}"
    )

    # Pre-load all models
    try:
        load_models()
        logger.info("All models loaded and ready")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Whisper Diarization API server...")


# Create FastAPI app
app = FastAPI(
    title="Whisper Diarization API",
    description="OpenAI-compatible API for audio transcription with speaker diarization",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(audio.router)


@app.get("/health-check")
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "asr_backend": settings.asr_backend,
        "model": settings.asr_model,
        "device": settings.asr_device,
        "compute_type": settings.asr_compute_type,
        "diarization_enabled": True,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)."""
    settings = get_settings()
    return {
        "object": "list",
        "data": [
            {
                "id": settings.asr_model,
                "object": "model",
                "created": 1677610602,
                "owned_by": "whisper-diarization",
            },
        ],
    }
