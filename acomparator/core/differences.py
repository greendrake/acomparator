"""Difference region detection from similarity timeline."""

import numpy as np


def find_difference_regions(
    similarity: np.ndarray,
    threshold: float,
    hop_length: int,
    sample_rate: int,
    min_duration: float,
) -> list[tuple[float, float]]:
    """
    Find contiguous regions where similarity drops below threshold.

    Args:
        similarity: Array of per-frame similarity scores.
        threshold: Similarity threshold below which frames are considered different.
        hop_length: Hop length used for frame extraction.
        sample_rate: Sample rate of the audio.
        min_duration: Minimum duration in seconds for a region to be reported.

    Returns:
        List of (start_seconds, end_seconds) tuples for each difference region.
    """
    if len(similarity) == 0:
        return []

    # Binary mask of frames below threshold
    below_threshold = similarity < threshold

    # Find contiguous runs
    regions = _find_contiguous_runs(below_threshold)

    # Convert frame indices to timestamps
    frame_duration = hop_length / sample_rate
    timed_regions = [
        (start * frame_duration, (end + 1) * frame_duration) for start, end in regions
    ]

    # Merge adjacent regions (separated by less than min_duration)
    merged = _merge_close_regions(timed_regions, min_duration)

    # Split large regions at significant similarity discontinuities
    split_regions = []
    for start, end in merged:
        if end - start > 10.0:  # Only split regions longer than 10 seconds
            sub_regions = _split_at_discontinuities(
                similarity, start, end, frame_duration, min_duration
            )
            split_regions.extend(sub_regions)
        else:
            split_regions.append((start, end))

    # Filter by minimum duration
    filtered = [
        (start, end) for start, end in split_regions if (end - start) >= min_duration
    ]

    return filtered


def _find_contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Find contiguous runs of True values in a boolean mask.

    Returns list of (start_index, end_index) tuples (inclusive).
    """
    if len(mask) == 0:
        return []

    runs = []
    in_run = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_run:
            # Start of a new run
            in_run = True
            start = i
        elif not val and in_run:
            # End of current run
            in_run = False
            runs.append((start, i - 1))

    # Handle run that extends to end
    if in_run:
        runs.append((start, len(mask) - 1))

    return runs


def _merge_close_regions(
    regions: list[tuple[float, float]], gap_threshold: float
) -> list[tuple[float, float]]:
    """
    Merge regions that are separated by less than gap_threshold.
    """
    if not regions:
        return []

    merged = [regions[0]]

    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]

        if start - prev_end < gap_threshold:
            # Merge with previous region
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def _split_at_discontinuities(
    similarity: np.ndarray,
    start_time: float,
    end_time: float,
    frame_duration: float,
    min_duration: float,
    window_seconds: float = 5.0,
    level_threshold: float = 0.2,
) -> list[tuple[float, float]]:
    """
    Split a difference region at points where similarity level changes significantly.

    This helps separate encoding differences (moderate similarity ~0.7) from
    content differences (low similarity ~0.5) that may have been merged together.

    Args:
        similarity: Full similarity array.
        start_time: Region start in seconds.
        end_time: Region end in seconds.
        frame_duration: Duration of each frame in seconds.
        min_duration: Minimum sub-region duration.
        window_seconds: Window size for computing local mean similarity.
        level_threshold: Minimum difference in mean levels to trigger a split.

    Returns:
        List of (start, end) tuples for sub-regions.
    """
    start_frame = int(start_time / frame_duration)
    end_frame = int(end_time / frame_duration)
    end_frame = min(end_frame, len(similarity))

    if end_frame <= start_frame:
        return [(start_time, end_time)]

    region_sim = similarity[start_frame:end_frame]
    window_frames = int(window_seconds / frame_duration)

    if len(region_sim) < window_frames * 2:
        return [(start_time, end_time)]

    # Compute windowed mean similarity
    step = window_frames // 2
    windowed_means = []
    for i in range(0, len(region_sim) - window_frames + 1, step):
        window = region_sim[i : i + window_frames]
        windowed_means.append((i, np.mean(window)))

    if len(windowed_means) < 4:
        return [(start_time, end_time)]

    # Find split points by comparing running max of previous windows to current
    # This catches gradual transitions that span multiple windows
    split_frames = []
    lookback = 3  # Compare to max of previous 3 windows

    for i in range(lookback, len(windowed_means)):
        # Get max mean from previous windows
        prev_max = max(wm[1] for wm in windowed_means[i - lookback : i])
        curr_mean = windowed_means[i][1]

        # If current is significantly lower than recent max, this is a transition
        # Check if this is the start of the low region (not already in it)
        if (
            prev_max - curr_mean > level_threshold
            and i > 0
            and windowed_means[i - 1][1] > curr_mean + 0.05
        ):
            split_frames.append(windowed_means[i][0])

    if not split_frames:
        return [(start_time, end_time)]

    # Deduplicate nearby split points (keep first of each cluster)
    min_gap_frames = int(10.0 / frame_duration)  # 10 second minimum between splits
    deduped = [split_frames[0]]
    for frame in split_frames[1:]:
        if frame - deduped[-1] > min_gap_frames:
            deduped.append(frame)
    split_frames = deduped

    # Create sub-regions
    sub_regions = []
    prev_frame = 0
    for split_frame in split_frames:
        sub_start = start_time + prev_frame * frame_duration
        sub_end = start_time + split_frame * frame_duration
        if sub_end - sub_start >= min_duration:
            sub_regions.append((sub_start, sub_end))
        prev_frame = split_frame

    # Add final sub-region
    final_start = start_time + prev_frame * frame_duration
    if end_time - final_start >= min_duration:
        sub_regions.append((final_start, end_time))

    return sub_regions if sub_regions else [(start_time, end_time)]
