"""Cut detection for finding removed sections in audio."""

import numpy as np


def detect_cut_region(
    similarity: np.ndarray,
    duration_diff: float,
    hop_length: int,
    sample_rate: int,
    window_frames: int = 20,
    threshold: float = 0.9,
) -> tuple[float, float] | None:
    """
    Detect a cut region by finding where similarity drops and stays low.

    When one file has a section cut out, the similarity will be high until
    the cut point, then drop because the content is offset.

    Args:
        similarity: Array of per-frame similarity scores.
        duration_diff: Difference in duration between the two files (seconds).
        hop_length: Hop length used for frame extraction.
        sample_rate: Sample rate of the audio.
        window_frames: Number of frames for sliding window average.
        threshold: Similarity threshold for detecting drop.

    Returns:
        Tuple of (cut_start_seconds, cut_end_seconds) or None if no cut detected.
    """
    if len(similarity) < window_frames or duration_diff < 1.0:
        return None

    frame_duration = hop_length / sample_rate

    # Find the first point where sustained similarity drops below threshold
    cut_frame = None
    for i in range(len(similarity) - window_frames):
        window_avg = np.mean(similarity[i : i + window_frames])
        if window_avg < threshold:
            cut_frame = i
            break

    if cut_frame is None:
        return None

    cut_start = cut_frame * frame_duration
    cut_end = cut_start + duration_diff

    return (cut_start, cut_end)
