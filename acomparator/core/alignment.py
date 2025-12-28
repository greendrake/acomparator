"""Audio alignment via cross-correlation on spectral features."""

import librosa
import numpy as np
from scipy import signal

ALIGNMENT_SAMPLE_RATE = 22050  # Sample rate for alignment
ALIGNMENT_HOP_LENGTH = 512  # Hop length for chroma extraction
REFINEMENT_WINDOW_SECONDS = 0.1  # Window to search for fine alignment


def find_alignment_offset(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: int,
) -> tuple[float, float]:
    """
    Find time offset between two audio signals using two-stage alignment.

    Stage 1: Coarse alignment using chroma features (robust to EQ/compression)
    Stage 2: Fine alignment using waveform cross-correlation (sample-accurate)

    Args:
        audio_a: First audio signal.
        audio_b: Second audio signal.
        sample_rate: Sample rate of both signals.

    Returns:
        Tuple of (offset_seconds, confidence).
        Positive offset means audio_b starts later than audio_a.
        Confidence is normalized correlation peak height in range [0, 1].
    """
    # Stage 1: Coarse alignment with chroma features
    coarse_offset, confidence = _coarse_alignment(audio_a, audio_b, sample_rate)

    # Stage 2: Refine with waveform cross-correlation
    refined_offset = _refine_alignment(audio_a, audio_b, sample_rate, coarse_offset)

    return refined_offset, confidence


def _coarse_alignment(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: int,
) -> tuple[float, float]:
    """Coarse alignment using chroma features."""
    # Extract chroma features - robust to EQ/compression differences
    chroma_a = librosa.feature.chroma_cqt(
        y=audio_a, sr=sample_rate, hop_length=ALIGNMENT_HOP_LENGTH
    )
    chroma_b = librosa.feature.chroma_cqt(
        y=audio_b, sr=sample_rate, hop_length=ALIGNMENT_HOP_LENGTH
    )

    # Flatten chroma to 1D by summing across pitch classes (energy envelope)
    energy_a = np.sum(chroma_a, axis=0)
    energy_b = np.sum(chroma_b, axis=0)

    # Normalize
    energy_a = (energy_a - np.mean(energy_a)) / (np.std(energy_a) + 1e-10)
    energy_b = (energy_b - np.mean(energy_b)) / (np.std(energy_b) + 1e-10)

    # Cross-correlation
    correlation = signal.correlate(energy_a, energy_b, mode="full", method="fft")

    # Find peak
    peak_idx = np.argmax(correlation)
    peak_value = correlation[peak_idx]

    # Convert index to offset in frames, then seconds
    zero_lag_idx = len(energy_b) - 1
    lag_frames = peak_idx - zero_lag_idx
    offset_seconds = lag_frames * ALIGNMENT_HOP_LENGTH / sample_rate

    # Confidence: peak value relative to theoretical max (len of shorter signal)
    max_possible = min(len(energy_a), len(energy_b))
    confidence = peak_value / max_possible if max_possible > 0 else 0.0

    return offset_seconds, float(np.clip(confidence, 0, 1))


def _refine_alignment(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: int,
    coarse_offset: float,
) -> float:
    """
    Refine alignment using waveform cross-correlation around the coarse offset.

    Searches within a small window around the coarse offset to find the
    sample-accurate best alignment.
    """
    coarse_samples = int(coarse_offset * sample_rate)
    window_samples = int(REFINEMENT_WINDOW_SECONDS * sample_rate)

    # Use first 5 seconds of audio for refinement (after alignment)
    ref_len = int(5 * sample_rate)

    # Determine aligned positions based on coarse offset
    if coarse_samples >= 0:
        # B starts later, trim A's start
        ref_start_a = coarse_samples
        ref_start_b = 0
    else:
        # A starts later, trim B's start
        ref_start_a = 0
        ref_start_b = -coarse_samples

    # Ensure we have enough audio
    if ref_start_a + ref_len > len(audio_a) or ref_start_b + ref_len > len(audio_b):
        ref_len = min(len(audio_a) - ref_start_a, len(audio_b) - ref_start_b)

    if ref_len < window_samples * 4:
        return coarse_offset

    # Extract reference from A at aligned position
    ref = audio_a[ref_start_a : ref_start_a + ref_len]

    # Extract target from B with extra samples on both sides to allow search
    target_start = max(0, ref_start_b - window_samples)
    target_end = min(len(audio_b), ref_start_b + ref_len + window_samples)
    target = audio_b[target_start:target_end]

    # How much padding we actually got on the left side
    left_padding = ref_start_b - target_start

    if len(target) < len(ref):
        return coarse_offset

    # Cross-correlate
    correlation = signal.correlate(target, ref, mode="valid")

    if len(correlation) == 0:
        return coarse_offset

    peak_idx = np.argmax(correlation)

    # peak_idx tells us where in target the ref pattern starts
    # If peak_idx == left_padding, the coarse alignment is perfect
    # If peak_idx < left_padding, B aligns earlier than expected (less trimming needed)
    # If peak_idx > left_padding, B aligns later than expected (more trimming needed)
    adjustment_samples = peak_idx - left_padding

    # For negative offset (trim B), adjustment reduces the trim amount
    # offset = -trim, so: new_offset = old_offset - adjustment
    refined_samples = coarse_samples - adjustment_samples
    refined_offset = refined_samples / sample_rate

    return refined_offset


def align_signals(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    offset_seconds: float,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two audio signals based on computed offset.

    Args:
        audio_a: First audio signal.
        audio_b: Second audio signal.
        offset_seconds: Offset from find_alignment_offset.
        sample_rate: Sample rate of both signals.

    Returns:
        Tuple of (aligned_a, aligned_b) with matching start points.
        Signals may have different lengths if one is longer.
    """
    offset_samples = int(offset_seconds * sample_rate)

    if offset_samples > 0:
        # audio_b starts later, trim audio_a start
        if offset_samples < len(audio_a):
            aligned_a = audio_a[offset_samples:]
            aligned_b = audio_b
        else:
            aligned_a = audio_a
            aligned_b = audio_b
    elif offset_samples < 0:
        # audio_a starts later, trim audio_b start
        trim = abs(offset_samples)
        if trim < len(audio_b):
            aligned_a = audio_a
            aligned_b = audio_b[trim:]
        else:
            aligned_a = audio_a
            aligned_b = audio_b
    else:
        aligned_a = audio_a
        aligned_b = audio_b

    return aligned_a, aligned_b
