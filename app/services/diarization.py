"""
Diarization service using NeMo ClusteringDiarizer.
Adapted from https://github.com/MahmoudAshraf97/whisper-diarization
"""

import logging
import os
import tempfile
import json
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

# Global diarizer instance
_diarizer = None


class SimpleDiarizer:
    """NeMo-based Speaker Diarizer using ClusteringDiarizer."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._validate_nemo()

    def _validate_nemo(self):
        """Validate NeMo is available."""
        import sys
        import traceback

        logger.info("Starting NeMo validation...")
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            # Step-by-step import to find crash location
            logger.info("Step 1: Importing nemo.core...")
            sys.stdout.flush()
            import nemo.core

            logger.info("Step 2: Importing nemo.collections...")
            sys.stdout.flush()
            import nemo.collections

            logger.info("Step 3: Importing nemo.collections.asr...")
            sys.stdout.flush()
            import nemo.collections.asr

            logger.info("Step 4: Importing nemo.collections.asr.models...")
            sys.stdout.flush()
            import nemo.collections.asr.models

            logger.info("Step 5: Accessing ClusteringDiarizer class...")
            sys.stdout.flush()
            from nemo.collections.asr.models import ClusteringDiarizer

            logger.info("NeMo ClusteringDiarizer imported successfully")
            sys.stdout.flush()
        except ImportError as e:
            logger.error(f"Failed to import NeMo: {e}")
            logger.error(traceback.format_exc())
            sys.stdout.flush()
            raise RuntimeError(
                "NeMo toolkit is required for diarization. "
                "Install with: pip install nemo_toolkit[asr]"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during NeMo import: {type(e).__name__}: {e}"
            )
            logger.error(traceback.format_exc())
            sys.stdout.flush()
            raise

    def _create_config(self, manifest_path: str, out_dir: str) -> str:
        """Create diarization config YAML."""
        config_yaml = f"""
name: ClusteringDiarizer
num_workers: 0
sample_rate: 16000
batch_size: 64
device: {self.device}
verbose: True

diarizer:
  manifest_filepath: {manifest_path}
  out_dir: {out_dir}
  oracle_vad: False
  collar: 0.1
  ignore_overlap: False
  
  vad:
    model_path: vad_multilingual_marblenet
    external_vad_manifest: null
    parameters:
      window_length_in_sec: 0.15
      shift_length_in_sec: 0.01
      smoothing: True
      overlap: 0.5
      onset: 0.4
      offset: 0.2
      pad_onset: 0.2
      pad_offset: 0.1
      min_duration_on: 0.1
      min_duration_off: 0.3
      filter_speech_first: True
  
  speaker_embeddings:
    model_path: titanet_large
    parameters:
      window_length_in_sec: [1.5, 1.0, 0.5]
      shift_length_in_sec: [0.75, 0.5, 0.25]
      multiscale_weights: [0.4, 0.35, 0.25]
      save_embeddings: True
  
  clustering:
    parameters:
      oracle_num_speakers: False
      max_num_speakers: 8
      enhanced_count_thres: 10
      max_rp_threshold: 0.3
      sparse_search_volume: 100
      maj_vote_spk_count: True
"""
        return config_yaml

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
        import soundfile as sf
        import numpy as np
        from nemo.collections.asr.models import ClusteringDiarizer
        from omegaconf import OmegaConf

        # Ensure correct shape
        if audio_waveform.dim() == 1:
            audio_waveform = audio_waveform.unsqueeze(0)

        # Create a persistent temp directory (not using context manager)
        temp_dir = tempfile.mkdtemp()

        try:
            # Save audio to temporary WAV file using soundfile
            temp_audio_path = os.path.join(temp_dir, "audio.wav")
            audio_np = audio_waveform.cpu().numpy().squeeze()
            sf.write(temp_audio_path, audio_np, sample_rate)

            # Create manifest file
            manifest_path = os.path.join(temp_dir, "manifest.json")
            duration = audio_waveform.shape[1] / sample_rate
            with open(manifest_path, "w") as f:
                manifest_entry = {
                    "audio_filepath": temp_audio_path,
                    "offset": 0,
                    "duration": duration,
                    "label": "infer",
                    "text": "-",
                    "num_speakers": None,
                    "rttm_filepath": None,
                    "uem_filepath": None,
                }
                f.write(json.dumps(manifest_entry) + "\n")

            # Create config with paths
            config_yaml = self._create_config(manifest_path, temp_dir)
            config_path = os.path.join(temp_dir, "diar_config.yaml")
            with open(config_path, "w") as f:
                f.write(config_yaml)

            # Load config and create diarizer
            config = OmegaConf.load(config_path)

            logger.info("Creating NeMo ClusteringDiarizer...")
            diarizer = ClusteringDiarizer(cfg=config)

            # Run diarization
            logger.info("Starting speaker diarization...")
            diarizer.diarize()
            logger.info("Speaker diarization complete")

            # Parse RTTM output
            rttm_path = os.path.join(temp_dir, "pred_rttms", "audio.rttm")
            if os.path.exists(rttm_path):
                speaker_ts = self._parse_rttm(rttm_path)
                logger.info(f"Parsed {len(speaker_ts)} speaker segments from RTTM")
            else:
                # Check alternative locations
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(".rttm"):
                            rttm_path = os.path.join(root, file)
                            speaker_ts = self._parse_rttm(rttm_path)
                            logger.info(
                                f"Found RTTM at {rttm_path}, parsed {len(speaker_ts)} segments"
                            )
                            break
                    else:
                        continue
                    break
                else:
                    logger.warning("No RTTM output found, returning empty speaker list")
                    speaker_ts = []

            # Clean up diarizer
            del diarizer
            if self.device == "cuda":
                torch.cuda.empty_cache()

        finally:
            # Clean up temp directory
            import shutil

            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir: {e}")

        return speaker_ts

    def _parse_rttm(self, rttm_path: str) -> List[Tuple[float, float, str]]:
        """Parse RTTM file to get speaker timestamps."""
        speaker_ts = []
        with open(rttm_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start = float(parts[3]) * 1000  # Convert to ms
                    duration = float(parts[4]) * 1000
                    speaker = parts[7]
                    speaker_ts.append((start, start + duration, speaker))

        # Sort by start time
        speaker_ts.sort(key=lambda x: x[0])
        return speaker_ts


def get_diarizer(device: str = "cuda") -> SimpleDiarizer:
    """Get or create the global diarizer instance."""
    global _diarizer
    if _diarizer is None:
        _diarizer = SimpleDiarizer(device=device)
    return _diarizer


def load_diarizer(device: str = "cuda") -> SimpleDiarizer:
    """Load and cache the diarizer."""
    return get_diarizer(device)
