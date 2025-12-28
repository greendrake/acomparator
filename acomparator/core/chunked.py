"""Chunked audio processing for memory-efficient comparison of large files."""

from collections.abc import Iterator
from pathlib import Path

import librosa
import numpy as np

from acomparator.core.loader import AudioLoadError

# Default chunk configuration
CHUNK_DURATION_SECONDS = 60
CHUNK_OVERLAP_SECONDS = 5
ALIGNMENT_SAMPLE_SECONDS = 120  # Use first N seconds for alignment


def get_audio_duration_samples(path: Path, sample_rate: int) -> tuple[float, int]:
    """
    Get audio duration without loading the full file.

    Returns:
        Tuple of (duration_seconds, total_samples).
    """
    duration = librosa.get_duration(path=path)
    total_samples = int(duration * sample_rate)
    return duration, total_samples


def load_audio_chunk(
    path: Path,
    sample_rate: int,
    offset_seconds: float,
    duration_seconds: float,
) -> np.ndarray:
    """
    Load a chunk of audio from a file.

    Args:
        path: Path to audio file.
        sample_rate: Target sample rate.
        offset_seconds: Start position in seconds.
        duration_seconds: Duration to load in seconds.

    Returns:
        Audio samples as float32 array, peak-normalized.
    """
    try:
        audio, _ = librosa.load(
            path,
            sr=sample_rate,
            mono=True,
            offset=offset_seconds,
            duration=duration_seconds,
        )
    except Exception as e:
        raise AudioLoadError(f"Failed to load chunk from {path}: {e}") from e

    # Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio


def iterate_audio_chunks(
    path: Path,
    sample_rate: int,
    chunk_duration: float = CHUNK_DURATION_SECONDS,
    overlap: float = CHUNK_OVERLAP_SECONDS,
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Iterate over audio file in chunks.

    Args:
        path: Path to audio file.
        sample_rate: Target sample rate.
        chunk_duration: Duration of each chunk in seconds.
        overlap: Overlap between chunks in seconds.

    Yields:
        Tuple of (chunk_start_seconds, chunk_audio).
    """
    total_duration, _ = get_audio_duration_samples(path, sample_rate)
    step = chunk_duration - overlap
    offset = 0.0

    while offset < total_duration:
        # Adjust last chunk to not exceed file duration
        remaining = total_duration - offset
        current_duration = min(chunk_duration, remaining)

        if current_duration < overlap:
            # Skip tiny final chunks
            break

        chunk = load_audio_chunk(path, sample_rate, offset, current_duration)
        yield offset, chunk

        offset += step


def load_audio_for_alignment(
    path: Path,
    sample_rate: int,
    max_seconds: float = ALIGNMENT_SAMPLE_SECONDS,
) -> np.ndarray:
    """
    Load audio sample for alignment (first N seconds only).

    For alignment, we don't need the entire file - the first couple
    minutes are sufficient to find the offset.
    """
    return load_audio_chunk(path, sample_rate, 0.0, max_seconds)
