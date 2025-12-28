"""Audio segment extraction using librosa and soundfile."""

from pathlib import Path

import librosa
import soundfile as sf


class ExtractionError(Exception):
    """Raised when segment extraction fails."""


def extract_segment(
    audio_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_path: Path,
) -> Path:
    """
    Extract a segment from an audio file and save it.

    Args:
        audio_path: Path to source audio file.
        start_seconds: Start time in seconds.
        end_seconds: End time in seconds.
        output_path: Path to save extracted segment.

    Returns:
        Path to the saved segment file.

    Raises:
        ExtractionError: If extraction fails.
    """
    try:
        duration = end_seconds - start_seconds
        if duration <= 0:
            raise ExtractionError(
                f"Invalid segment range: {start_seconds}s - {end_seconds}s"
            )

        # Load only the required segment
        audio, sr = librosa.load(
            audio_path,
            sr=None,  # Preserve original sample rate
            mono=False,  # Preserve channels
            offset=start_seconds,
            duration=duration,
        )
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"Failed to load audio file {audio_path}: {e}") from e

    if audio.size == 0:
        raise ExtractionError(
            f"Invalid segment range: {start_seconds}s - {end_seconds}s "
            f"(segment is empty)"
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # soundfile expects shape (samples, channels) for multi-channel
        # librosa returns (channels, samples) for multi-channel or (samples,) for mono
        if audio.ndim == 2:
            audio = audio.T  # Transpose to (samples, channels)
        sf.write(output_path, audio, sr)
    except Exception as e:
        raise ExtractionError(f"Failed to export segment to {output_path}: {e}") from e

    return output_path


def get_audio_duration(audio_path: Path) -> float:
    """
    Get duration of an audio file in seconds.

    Args:
        audio_path: Path to audio file.

    Returns:
        Duration in seconds.

    Raises:
        ExtractionError: If file cannot be loaded.
    """
    try:
        return librosa.get_duration(path=audio_path)
    except Exception as e:
        raise ExtractionError(f"Failed to get duration of {audio_path}: {e}") from e
