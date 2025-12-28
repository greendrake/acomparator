"""Audio loading and normalization."""

from pathlib import Path

import librosa
import numpy as np


class AudioLoadError(Exception):
    """Raised when audio file cannot be loaded."""


def load_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    """
    Load audio file, convert to mono, resample, and normalize amplitude.

    Args:
        path: Path to audio file.
        sample_rate: Target sample rate for resampling.

    Returns:
        Tuple of (audio samples as float32 array, sample rate).

    Raises:
        AudioLoadError: If file cannot be loaded.
    """
    if not path.exists():
        raise AudioLoadError(f"File not found: {path}")

    try:
        audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio file {path}: {e}") from e

    # Peak normalization to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio, sr
