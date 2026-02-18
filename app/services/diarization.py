"""
Diarization service using NeMo NeuralDiarizer (MSDD).
Adapted from https://github.com/MahmoudAshraf97/whisper-diarization
"""

import json
import logging
import os
import tempfile
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

# Global diarizer instance
_diarizer = None


class MSDDDiarizer:
    """NeMo-based Speaker Diarizer using NeuralDiarizer (MSDD)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._validate_nemo()
        self._initialize_model()

    def _validate_nemo(self):
        """Validate NeMo is available."""
        import sys

        logger.info("Starting NeMo validation...")
        sys.stdout.flush()

        try:
            logger.info("Importing nemo.collections.asr.models...")
            sys.stdout.flush()
            from nemo.collections.asr.models.msdd_models import NeuralDiarizer

            logger.info("NeMo NeuralDiarizer imported successfully")
            sys.stdout.flush()
        except ImportError as e:
            logger.error(f"Failed to import NeMo: {e}")
            raise RuntimeError(
                "NeMo toolkit is required for diarization. "
                "Install with: pip install nemo_toolkit[asr]"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during NeMo import: {type(e).__name__}: {e}"
            )
            raise

    def _initialize_model(self):
        """Initialize the NeuralDiarizer model."""
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer
        from omegaconf import OmegaConf

        # Load base config from YAML
        config_path = os.path.join(
            os.path.dirname(__file__), "diar_infer_telephonic.yaml"
        )
        config = OmegaConf.load(config_path)

        # Apply runtime overrides
        config.diarizer.out_dir = None
        config.diarizer.manifest_filepath = None
        config.diarizer.speaker_embeddings.model_path = "titanet_large"
        config.diarizer.oracle_vad = False
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        config.diarizer.vad.model_path = "vad_multilingual_marblenet"
        config.diarizer.vad.parameters.onset = 0.8
        config.diarizer.vad.parameters.offset = 0.6
        config.diarizer.vad.parameters.pad_offset = -0.05
        config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"

        logger.info("Creating NeMo NeuralDiarizer...")
        self.model = NeuralDiarizer(cfg=config).to(self.device)
        logger.info("NeuralDiarizer initialized successfully")

    def diarize(
        self, audio_waveform: torch.Tensor, sample_rate: int = 16000
    ) -> List[Tuple[float, float, str]]:
        """
        Perform speaker diarization on audio waveform.

        Args:
            audio_waveform: Audio tensor of shape (1, samples) or (samples,)
            sample_rate: Audio sample rate (default 16000)

        Returns:
            List of (start_ms, end_ms, speaker_id) tuples
        """
        import torchaudio
        from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels

        # Ensure correct shape: (1, samples)
        if audio_waveform.dim() == 1:
            audio_waveform = audio_waveform.unsqueeze(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save audio to temporary WAV file
            temp_audio_path = os.path.join(temp_dir, "mono_file.wav")
            torchaudio.save(
                temp_audio_path,
                audio_waveform,
                sample_rate,
                channels_first=True,
            )

            # Create manifest file
            manifest_path = os.path.join(temp_dir, "manifest.json")
            meta = {
                "audio_filepath": temp_audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "rttm_filepath": None,
                "uem_filepath": None,
            }
            with open(manifest_path, "w") as f:
                json.dump(meta, f)

            # Configure and run diarization
            logger.info("Starting speaker diarization (MSDD)...")
            self.model._initialize_configs(
                manifest_path=manifest_path,
                max_speakers=8,
                num_speakers=None,
                tmpdir=temp_dir,
                batch_size=24,
                num_workers=0,
                verbose=True,
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.out_dir = (
                temp_dir
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.manifest_filepath = (
                manifest_path
            )
            self.model.msdd_model.cfg.test_ds.manifest_filepath = manifest_path
            self.model.diarize()
            logger.info("Speaker diarization complete")

            # Parse RTTM output
            rttm_path = os.path.join(temp_dir, "pred_rttms", "mono_file.rttm")
            pred_labels = rttm_to_labels(rttm_path)

            speaker_ts = []
            for label in pred_labels:
                start, end, speaker = label.split()
                start, end = float(start), float(end)
                start_ms, end_ms = int(start * 1000), int(end * 1000)
                speaker_id = speaker.split("_")[1]
                speaker_ts.append((start_ms, end_ms, speaker_id))

            speaker_ts.sort(key=lambda x: x[0])
            logger.info(f"Parsed {len(speaker_ts)} speaker segments")

        return speaker_ts


def get_diarizer(device: str = "cuda") -> MSDDDiarizer:
    """Get or create the global diarizer instance."""
    global _diarizer
    if _diarizer is None:
        _diarizer = MSDDDiarizer(device=device)
    return _diarizer


def load_diarizer(device: str = "cuda") -> MSDDDiarizer:
    """Load and cache the diarizer."""
    return get_diarizer(device)
