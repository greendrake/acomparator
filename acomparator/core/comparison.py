"""Main comparison pipeline orchestrating all components."""

from enum import Enum, auto
from pathlib import Path

import numpy as np

from acomparator.config import Config
from acomparator.core.alignment import find_alignment_offset
from acomparator.core.chunked import (
    CHUNK_DURATION_SECONDS,
    CHUNK_OVERLAP_SECONDS,
    get_audio_duration_samples,
    load_audio_chunk,
    load_audio_for_alignment,
)
from acomparator.core.cuts import detect_cut_region
from acomparator.core.differences import find_difference_regions
from acomparator.core.fingerprint import compare_fingerprints, compute_fingerprint
from acomparator.core.similarity import MFCCCosineSimilarity
from acomparator.extraction.segments import extract_segment, get_audio_duration
from acomparator.models import ComparisonReport, ComparisonResult, DifferenceSection

# Threshold for considering alignment valid
ALIGNMENT_CONFIDENCE_THRESHOLD = 0.3

# Target sample rate for difference analysis (reduces memory usage)
ANALYSIS_SAMPLE_RATE = 22050


class DifferenceClass(Enum):
    """Classification of a difference region."""

    AUDIBLE = auto()  # Real content difference - report it
    PHASE_INVERSION = auto()  # Phase inverted - report it
    ENCODING = auto()  # Encoding/mastering artifact - filter out
    GAIN = auto()  # Level difference only - filter out
    SILENCE = auto()  # Trailing/leading silence - filter out


# =============================================================================
# Audio Analysis Helpers
# =============================================================================


def _calculate_effective_starts(offset_seconds: float) -> tuple[float, float]:
    """Calculate effective start times for aligned comparison.

    Returns:
        Tuple of (effective_start_a, effective_start_b).
    """
    if offset_seconds > 0:
        return offset_seconds, 0.0
    else:
        return 0.0, abs(offset_seconds)


def _compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (root mean square) level of audio signal."""
    return float(np.sqrt(np.mean(audio**2)))


def _load_segment_for_analysis(
    path: Path,
    start_seconds: float,
    end_seconds: float,
    target_sr: int = ANALYSIS_SAMPLE_RATE,
) -> np.ndarray:
    """Load a segment from an audio file for analysis.

    Uses ffmpeg to extract the segment efficiently without loading the whole file.

    Returns:
        Audio samples at target sample rate.
    """
    import io
    import subprocess

    import soundfile as sf

    duration = end_seconds - start_seconds
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_seconds),
            "-t",
            str(duration),
            "-i",
            str(path),
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    audio, _ = sf.read(io.BytesIO(result.stdout))
    return audio


def _classify_audio_pair(
    a: np.ndarray, b: np.ndarray, sr: int = ANALYSIS_SAMPLE_RATE
) -> DifferenceClass:
    """
    Classify the difference between two audio segments.

    Args:
        a: First audio segment (mono, resampled).
        b: Second audio segment (mono, resampled).
        sr: Sample rate of the audio.

    Returns:
        DifferenceClass indicating the type of difference.
    """
    import librosa

    # Need same length for comparison
    min_len = min(len(a), len(b))
    if min_len < 100:
        return DifferenceClass.ENCODING

    a = a[:min_len]
    b = b[:min_len]

    # For very short segments (< 0.2s), MFCC/RMS analysis is unreliable
    min_reliable_samples = int(0.2 * sr)
    is_very_short = min_len < min_reliable_samples

    # Check RMS levels
    rms_a = _compute_rms(a)
    rms_b = _compute_rms(b)

    if rms_a < 1e-6 and rms_b < 1e-6:
        return DifferenceClass.SILENCE

    if rms_a < 1e-4 and rms_b < 1e-4:
        return DifferenceClass.SILENCE

    if rms_a < 1e-6 or rms_b < 1e-6:
        # One segment is silent, other has content - audible
        return DifferenceClass.AUDIBLE

    # Compute waveform correlation
    avg_corr = _compute_waveform_correlation(a, b)
    if avg_corr is None:
        return DifferenceClass.AUDIBLE

    # Check for phase inversion (correlation ≈ -1)
    if avg_corr < -0.95:
        # Very short phase inversions (<0.15s) with high spectral similarity
        # are almost always encoding artifacts, not intentional edits
        # (different encoders/mastering can flip polarity in short segments)
        is_very_short_phase = min_len < int(0.15 * sr)
        if is_very_short_phase:
            try:
                mfcc_a = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=13)
                mfcc_b = librosa.feature.mfcc(y=b, sr=sr, n_mfcc=13)
                min_frames = min(mfcc_a.shape[1], mfcc_b.shape[1])
                if min_frames > 0:
                    mfcc_a_flat = mfcc_a[:, :min_frames].flatten()
                    mfcc_b_flat = mfcc_b[:, :min_frames].flatten()
                    mfcc_corr = np.corrcoef(mfcc_a_flat, mfcc_b_flat)[0, 1]
                    if mfcc_corr > 0.90:
                        return DifferenceClass.ENCODING
            except Exception:
                pass
        return DifferenceClass.PHASE_INVERSION

    # Check for gain difference with otherwise identical content
    if avg_corr > 0.95:
        rms_ratio = rms_a / rms_b
        if rms_ratio < 0.9 or rms_ratio > 1.1:
            return DifferenceClass.GAIN

    # Check if it's encoding/mastering difference
    if avg_corr < 0.95:
        # Very short segments are almost always encoding artifacts
        if is_very_short and avg_corr > -0.9:
            return DifferenceClass.ENCODING

        try:
            mfcc_a = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=13)
            mfcc_b = librosa.feature.mfcc(y=b, sr=sr, n_mfcc=13)
            min_frames = min(mfcc_a.shape[1], mfcc_b.shape[1])
            mfcc_a_flat = mfcc_a[:, :min_frames].flatten()
            mfcc_b_flat = mfcc_b[:, :min_frames].flatten()
            mfcc_corr = np.corrcoef(mfcc_a_flat, mfcc_b_flat)[0, 1]

            # For spectral similarity, check if mastering/encoding difference
            if mfcc_corr > 0.80:
                rms_a_env = librosa.feature.rms(y=a)[0]
                rms_b_env = librosa.feature.rms(y=b)[0]
                min_rms = min(len(rms_a_env), len(rms_b_env))
                if min_rms > 1:
                    rms_corr = np.corrcoef(rms_a_env[:min_rms], rms_b_env[:min_rms])[
                        0, 1
                    ]
                    if np.isnan(rms_corr):
                        rms_corr = 1.0
                    if rms_corr > 0.70:
                        # Check for significant level difference
                        rms_ratio = rms_a / rms_b
                        if rms_ratio < 0.7 or rms_ratio > 1.4:
                            return DifferenceClass.GAIN
                        return DifferenceClass.ENCODING
                else:
                    return DifferenceClass.ENCODING

            # For short segments with good spectral and envelope similarity,
            # likely mastering difference rather than intentional edit
            is_short = min_len < int(0.6 * sr)
            if is_short and mfcc_corr > 0.80:
                rms_a_env = librosa.feature.rms(y=a)[0]
                rms_b_env = librosa.feature.rms(y=b)[0]
                min_rms = min(len(rms_a_env), len(rms_b_env))
                if min_rms > 1:
                    rms_corr = np.corrcoef(rms_a_env[:min_rms], rms_b_env[:min_rms])[
                        0, 1
                    ]
                    if np.isnan(rms_corr):
                        rms_corr = 0.0
                    rms_ratio = rms_a / rms_b if rms_b > 1e-6 else 0

                    # Classify as encoding/mastering if:
                    # - RMS levels are similar (not silenced/heavily filtered)
                    # - Either envelope is similar OR MFCC is very high
                    #   (very high MFCC = same content, dynamics may differ)
                    if 0.5 < rms_ratio < 2.0 and (rms_corr > 0.5 or mfcc_corr > 0.95):
                        return DifferenceClass.ENCODING
        except Exception:
            pass

    return DifferenceClass.AUDIBLE


def _classify_single_audio(audio: np.ndarray) -> DifferenceClass:
    """Classify a single audio segment (trailing/leading content)."""
    rms = _compute_rms(audio)
    if rms < 1e-4:
        return DifferenceClass.SILENCE
    return DifferenceClass.AUDIBLE


def _classify_region(
    file_a: Path,
    file_b: Path,
    start: float,
    end: float,
    offset_seconds: float,
    duration_a: float,
    duration_b: float,
) -> DifferenceClass:
    """
    Classify a difference region by loading and analyzing audio from original files.

    Returns:
        DifferenceClass indicating the type of difference.
    """
    time_in_a_start = start + max(0, offset_seconds)
    time_in_a_end = end + max(0, offset_seconds)
    time_in_b_start = start + max(0, -offset_seconds)
    time_in_b_end = end + max(0, -offset_seconds)

    has_a = time_in_a_start < duration_a
    has_b = time_in_b_start < duration_b

    try:
        if has_a and has_b:
            # Both files have content in this region
            a = _load_segment_for_analysis(
                file_a, time_in_a_start, min(time_in_a_end, duration_a)
            )
            b = _load_segment_for_analysis(
                file_b, time_in_b_start, min(time_in_b_end, duration_b)
            )
            return _classify_audio_pair(a, b)
        elif has_a:
            # Only file A has content (trailing in A)
            a = _load_segment_for_analysis(
                file_a, time_in_a_start, min(time_in_a_end, duration_a)
            )
            return _classify_single_audio(a)
        elif has_b:
            # Only file B has content (trailing in B)
            b = _load_segment_for_analysis(
                file_b, time_in_b_start, min(time_in_b_end, duration_b)
            )
            return _classify_single_audio(b)
        else:
            # Neither has content - shouldn't happen
            return DifferenceClass.SILENCE
    except Exception:
        # If we can't load/analyze, assume audible to be safe
        return DifferenceClass.AUDIBLE


def _is_reportable(diff_class: DifferenceClass) -> bool:
    """Check if a difference class should be reported to the user."""
    return diff_class in (DifferenceClass.AUDIBLE, DifferenceClass.PHASE_INVERSION)


def _compute_waveform_correlation(
    a: np.ndarray, b: np.ndarray, chunk_size: int = 10000
) -> float | None:
    """Compute average waveform correlation between two signals.

    Processes in chunks to avoid memory issues with long signals.

    Returns:
        Average correlation coefficient, or None if computation fails.
    """
    min_len = min(len(a), len(b))
    chunk_size = min(chunk_size, min_len)

    correlations = []
    for i in range(0, min_len - chunk_size + 1, chunk_size):
        chunk_a = a[i : i + chunk_size]
        chunk_b = b[i : i + chunk_size]
        # Normalize chunks (zero mean)
        chunk_a = chunk_a - np.mean(chunk_a)
        chunk_b = chunk_b - np.mean(chunk_b)
        std_a = np.std(chunk_a)
        std_b = np.std(chunk_b)
        if std_a > 1e-6 and std_b > 1e-6:
            corr = np.sum(chunk_a * chunk_b) / (len(chunk_a) * std_a * std_b)
            correlations.append(corr)

    if not correlations:
        return None

    return float(np.mean(correlations))


def compare_audio_files(
    file_a: Path,
    file_b: Path,
    config: Config,
) -> ComparisonReport:
    """
    Compare two audio files and generate a comparison report.

    Uses chunked processing for memory efficiency with large files.

    Args:
        file_a: Path to first audio file.
        file_b: Path to second audio file.
        config: Comparison configuration.

    Returns:
        ComparisonReport with result classification and any differences.
    """
    sr = config.sample_rate

    # Step 1: Fingerprint check (if enabled)
    fingerprint_match: bool | None = None
    if config.use_fingerprint:
        fingerprint_match = _check_fingerprint_similarity(file_a, file_b)

    # Step 2: Get durations (without loading full files)
    duration_a, _ = get_audio_duration_samples(file_a, sr)
    duration_b, _ = get_audio_duration_samples(file_b, sr)
    duration_diff = abs(duration_a - duration_b)

    # Step 3: Load samples for alignment (first N seconds only)
    align_sample_a = load_audio_for_alignment(file_a, sr)
    align_sample_b = load_audio_for_alignment(file_b, sr)

    # Step 4: Find alignment offset
    offset_seconds, alignment_confidence = find_alignment_offset(
        align_sample_a, align_sample_b, sr
    )

    # Free alignment samples
    del align_sample_a, align_sample_b

    # Step 5: Check for completely different audio
    if alignment_confidence < ALIGNMENT_CONFIDENCE_THRESHOLD:
        return ComparisonReport(
            result=ComparisonResult.COMPLETELY_DIFFERENT,
            overall_similarity=0.0,
            alignment_offset_seconds=offset_seconds,
            fingerprint_match=fingerprint_match,
            differences=[],
        )

    # Step 6: Detect potential cut scenario
    # If fingerprints match but there's a significant duration difference,
    # this suggests a cut/edit rather than just a different starting point.
    # Cases:
    #   1. offset ≈ duration_diff: content cut from start, use detected offset
    #   2. offset ≈ 0 AND duration_diff large AND confidence low: cut elsewhere, use zero
    #
    # Important: Small sub-second offsets (e.g., 60ms from different masterings) should
    # NOT trigger zero offset - they are real alignment differences that must be applied.
    use_zero_offset = False
    if fingerprint_match and duration_diff > 1.0:
        # Only force zero offset if:
        # - Detected offset is very small (under 0.5s) AND
        # - Duration difference is large (over 10s) AND
        # - Alignment confidence suggests the offset is noise, not real alignment
        # This handles cuts from middle/end while preserving real sub-second offsets
        offset_explains_duration = abs(abs(offset_seconds) - duration_diff) < 2.0
        offset_is_noise = (
            abs(offset_seconds) < 0.5
            and duration_diff > 10.0
            and alignment_confidence < 0.8
        )
        if offset_is_noise and not offset_explains_duration:
            use_zero_offset = True
            offset_seconds = 0.0

    # Step 7: Compute similarity in chunks
    similarity = _compute_chunked_similarity(
        file_a,
        file_b,
        sr,
        config.hop_length,
        config.n_mfcc,
        offset_seconds,
        duration_a,
        duration_b,
    )

    # Step 8: Calculate overall similarity
    overall_similarity = float(np.mean(similarity))

    # Step 9: Find difference regions
    diff_regions: list[tuple[float, float]] = []

    # Always look for content differences in the overlapping region
    diff_regions = find_difference_regions(
        similarity,
        threshold=config.diff_threshold,
        hop_length=config.hop_length,
        sample_rate=sr,
        min_duration=config.min_diff_duration,
    )

    # For cut scenarios, also check for cut-specific patterns
    if use_zero_offset:
        cut_region = detect_cut_region(
            similarity,
            duration_diff=duration_diff,
            hop_length=config.hop_length,
            sample_rate=sr,
        )
        if cut_region:
            # Add cut region if not already covered by diff_regions
            overlaps = any(
                not (cut_region[1] < r[0] or cut_region[0] > r[1]) for r in diff_regions
            )
            if not overlaps:
                diff_regions.append(cut_region)

    # Always check for trailing content when durations differ by more than min_diff_duration
    # This catches extra content at the end that cut detection might miss
    trailing_region = _get_trailing_difference(
        duration_a, duration_b, offset_seconds, config.min_diff_duration
    )
    # Avoid duplicate if cut_region already covers this
    if trailing_region and (
        not diff_regions or trailing_region[0] > diff_regions[-1][1]
    ):
        diff_regions.append(trailing_region)

    # Step 10: Preliminary classification
    # For cut scenarios, if a cut was detected, always classify as partial match
    if use_zero_offset and diff_regions:
        result = ComparisonResult.PARTIAL_MATCH
    else:
        result = _classify_result(
            overall_similarity,
            diff_regions,
            config.same_threshold,
            config.different_threshold,
            fingerprint_match,
        )

    # Step 11: Classify and filter difference regions
    differences: list[DifferenceSection] = []
    if result == ComparisonResult.PARTIAL_MATCH and diff_regions:
        # Get file durations for classification
        duration_a_full = get_audio_duration(file_a)
        duration_b_full = get_audio_duration(file_b)

        # Classify each region and filter to only reportable ones
        reportable_regions: list[tuple[float, float]] = []
        for start, end in diff_regions:
            diff_class = _classify_region(
                file_a,
                file_b,
                start,
                end,
                offset_seconds,
                duration_a_full,
                duration_b_full,
            )
            if _is_reportable(diff_class):
                reportable_regions.append((start, end))

        # Only extract segments for reportable differences
        if reportable_regions:
            differences = _extract_differences(
                file_a, file_b, reportable_regions, offset_seconds, config.output_dir
            )

        # Re-classify if all differences were filtered out
        if not differences:
            result = ComparisonResult.SAME_SOURCE

    return ComparisonReport(
        result=result,
        overall_similarity=overall_similarity,
        alignment_offset_seconds=offset_seconds,
        fingerprint_match=fingerprint_match,
        differences=differences,
    )


def _compute_chunked_similarity(
    file_a: Path,
    file_b: Path,
    sample_rate: int,
    hop_length: int,
    n_mfcc: int,
    offset_seconds: float,
    duration_a: float,
    duration_b: float,
) -> np.ndarray:
    """
    Compute frame-wise similarity using chunked processing.

    Processes audio in chunks to avoid loading entire files into memory.
    """
    similarity_algo = MFCCCosineSimilarity(n_mfcc=n_mfcc)

    # Calculate aligned durations
    effective_start_a, effective_start_b = _calculate_effective_starts(offset_seconds)
    effective_duration_a = duration_a - effective_start_a
    effective_duration_b = duration_b - effective_start_b
    compare_duration = min(effective_duration_a, effective_duration_b)

    # Calculate expected number of frames
    total_samples = int(compare_duration * sample_rate)
    total_frames = total_samples // hop_length

    # Pre-allocate similarity array
    similarity = np.zeros(total_frames)

    chunk_duration = CHUNK_DURATION_SECONDS
    overlap = CHUNK_OVERLAP_SECONDS
    step = chunk_duration - overlap

    chunk_start = 0.0

    while chunk_start < compare_duration:
        # Calculate chunk boundaries
        remaining = compare_duration - chunk_start
        current_chunk_duration = min(chunk_duration, remaining)

        if current_chunk_duration < overlap:
            break

        # Load chunks from both files (with alignment offset applied)
        chunk_a = load_audio_chunk(
            file_a, sample_rate, effective_start_a + chunk_start, current_chunk_duration
        )
        chunk_b = load_audio_chunk(
            file_b, sample_rate, effective_start_b + chunk_start, current_chunk_duration
        )

        # Compute similarity for this chunk
        chunk_similarity = similarity_algo.compute_frame_similarity(
            chunk_a, chunk_b, sample_rate, hop_length
        )

        # Calculate frame positions based on actual time (not accumulated indices)
        # This prevents drift over many chunks
        if chunk_start == 0.0:
            # First chunk: starts at frame 0, use all frames
            chunk_frame_start = 0
            src_start_frame = 0
        else:
            # Subsequent chunks: calculate position from time
            # The usable portion starts after the overlap
            usable_time_start = chunk_start + overlap
            chunk_frame_start = int(usable_time_start * sample_rate / hop_length)
            # Skip overlap frames in the chunk similarity array
            src_start_frame = int(overlap * sample_rate / hop_length)

        # Calculate how many frames to copy
        src_frames_available = len(chunk_similarity) - src_start_frame
        dst_frames_available = total_frames - chunk_frame_start
        frames_to_copy = min(src_frames_available, dst_frames_available)

        if frames_to_copy > 0:
            similarity[chunk_frame_start : chunk_frame_start + frames_to_copy] = (
                chunk_similarity[src_start_frame : src_start_frame + frames_to_copy]
            )

        chunk_start += step

        # Free chunk memory
        del chunk_a, chunk_b, chunk_similarity

    return similarity


def _check_fingerprint_similarity(file_a: Path, file_b: Path) -> bool:
    """Check if fingerprints indicate same source."""
    _, fp_a = compute_fingerprint(file_a)
    _, fp_b = compute_fingerprint(file_b)
    similarity = compare_fingerprints(fp_a, fp_b)
    return similarity > 0.7


def _get_trailing_difference(
    duration_a: float,
    duration_b: float,
    offset_seconds: float,
    min_duration: float,
) -> tuple[float, float] | None:
    """Detect trailing content difference due to duration mismatch."""
    effective_start_a, effective_start_b = _calculate_effective_starts(offset_seconds)
    effective_duration_a = duration_a - effective_start_a
    effective_duration_b = duration_b - effective_start_b
    overlap_duration = min(effective_duration_a, effective_duration_b)

    if effective_duration_a > effective_duration_b + min_duration:
        trailing_start = overlap_duration
        trailing_end = effective_duration_a
        if trailing_end - trailing_start >= min_duration:
            return (trailing_start, trailing_end)

    if effective_duration_b > effective_duration_a + min_duration:
        trailing_start = overlap_duration
        trailing_end = effective_duration_b
        if trailing_end - trailing_start >= min_duration:
            return (trailing_start, trailing_end)

    return None


def _classify_result(
    overall_similarity: float,
    diff_regions: list[tuple[float, float]],
    same_threshold: float,
    different_threshold: float,
    fingerprint_match: bool | None,
) -> ComparisonResult:
    """Classify comparison result based on similarity, differences, and fingerprint.

    Classification logic:
    1. Below different_threshold → completely_different
    2. No fingerprint match AND low similarity (< 0.5) → completely_different
       (handles reversed audio and other non-matching content)
    3. Difference regions detected → partial_match
    4. Above same_threshold → same_source
    """
    if overall_similarity < different_threshold:
        return ComparisonResult.COMPLETELY_DIFFERENT

    # If fingerprints don't match and similarity is below 0.5,
    # treat as completely different (e.g., reversed audio)
    if fingerprint_match is False and overall_similarity < 0.5:
        return ComparisonResult.COMPLETELY_DIFFERENT

    # If difference regions were detected, it's a partial match
    # even if overall similarity is high (small edit in long file)
    if diff_regions:
        return ComparisonResult.PARTIAL_MATCH

    if overall_similarity >= same_threshold:
        return ComparisonResult.SAME_SOURCE

    return ComparisonResult.SAME_SOURCE


def _extract_differences(
    file_a: Path,
    file_b: Path,
    regions: list[tuple[float, float]],
    offset_seconds: float,
    output_dir: Path,
) -> list[DifferenceSection]:
    """Extract audio segments for each difference region.

    Only call this for regions that have already been classified as reportable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    duration_a = get_audio_duration(file_a)
    duration_b = get_audio_duration(file_b)

    differences = []
    for idx, (start, end) in enumerate(regions):
        time_in_a_start = start + max(0, offset_seconds)
        time_in_a_end = end + max(0, offset_seconds)
        time_in_b_start = start + max(0, -offset_seconds)
        time_in_b_end = end + max(0, -offset_seconds)

        segment_a_path: Path | None = None
        segment_b_path: Path | None = None

        if time_in_a_start < duration_a:
            segment_a_path = output_dir / f"diff_{idx + 1:03d}_a.wav"
            extract_segment(
                file_a,
                time_in_a_start,
                min(time_in_a_end, duration_a),
                segment_a_path,
            )

        if time_in_b_start < duration_b:
            segment_b_path = output_dir / f"diff_{idx + 1:03d}_b.wav"
            extract_segment(
                file_b,
                time_in_b_start,
                min(time_in_b_end, duration_b),
                segment_b_path,
            )

        differences.append(
            DifferenceSection(
                start_seconds=start,
                end_seconds=end,
                duration_seconds=end - start,
                file_a_segment=segment_a_path,
                file_b_segment=segment_b_path,
                comment=None,
            )
        )

    return differences
