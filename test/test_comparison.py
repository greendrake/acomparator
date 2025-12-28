"""Tests for acomparator CLI tool."""

import json
import os
import subprocess
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def run_compare(file_a: str, file_b: str, **kwargs) -> dict:
    """Run acomparator CLI and return parsed JSON result.

    Uses ACOMPARATOR_EXECUTABLE env var if set, otherwise runs via Python module.
    """
    executable = os.environ.get("ACOMPARATOR_EXECUTABLE")
    if executable:
        cmd = [executable]
    else:
        cmd = ["python", "-m", "acomparator.cli"]

    cmd.extend([str(FIXTURES_DIR / file_a), str(FIXTURES_DIR / file_b)])
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return json.loads(result.stdout)


# =============================================================================
# 1. Same Source Tests
# =============================================================================


def test_identical_files():
    """1.1 - Two identical files should have near-perfect similarity."""
    result = run_compare("source.wav", "source_copy.wav")
    assert result["result"] == "same_source"
    assert result["overall_similarity"] > 0.99
    assert result["differences"] == []


def test_silence_prefix():
    """1.2 - Same source with silence prefix should detect offset."""
    result = run_compare("source.wav", "source_silence_prefix.wav")
    assert result["result"] == "same_source"
    assert abs(result["alignment_offset_seconds"]) > 2.5  # ~3s offset
    assert result["differences"] == []


def test_degraded_quality():
    """1.3 - Degraded quality (volume, noise) should still be same source."""
    result = run_compare("source.wav", "source_degraded.wav")
    assert result["result"] == "same_source"
    assert result["overall_similarity"] > 0.85
    assert result["differences"] == []


def test_trailing_content():
    """1.4 - File with extra trailing content should be partial_match.

    When one file has additional content at the end, this must be detected
    as a difference even when fingerprints match and overlapping content
    is identical.
    """
    result = run_compare("source.wav", "source_with_trailing.wav")
    assert result["result"] == "partial_match"
    assert result["fingerprint_match"] is True
    assert len(result["differences"]) == 1
    diff = result["differences"][0]
    # Source is ~37s, trailing content adds 5s starting at ~37s
    assert 36 < diff["start_seconds"] < 38
    assert 4.5 < diff["duration_seconds"] < 5.5
    # file_a has no content here, file_b has the trailing segment
    assert diff["file_a_segment"] is None
    assert diff["file_b_segment"] is not None


def test_encoding_difference_filtering():
    """1.5 - Encoding differences should be filtered, content differences detected.

    This tests a file with two types of differences:
    - 10-20s: Encoding difference (source + 5% noise) - perceptually similar
    - 20-25s: Content difference (reversed audio) - actually different

    Before the fix, both regions would be merged into one large difference
    starting at 10s. After the fix, only the content difference at 20-25s
    should be reported because encoding differences are filtered out.
    """
    result = run_compare("source.wav", "source_encoding_then_content_diff.wav")
    assert result["result"] == "partial_match"
    assert len(result["differences"]) == 1
    diff = result["differences"][0]
    # Only the content difference (20-25s) should be reported
    # The encoding difference (10-20s) should be filtered out
    assert 19.5 < diff["start_seconds"] < 21.0
    assert 4.0 < diff["duration_seconds"] < 6.0


def test_different_mastering_same_source():
    """1.6 - Different mastering with offset and trailing silence is same source.

    This tests a scenario common with different rips/masters of the same track:
    - 60ms start offset (one file starts slightly earlier)
    - Subtle encoding noise throughout (different codec artifacts)
    - 2s trailing silence at the end

    Before the fix:
    - Small offset was ignored due to cut detection logic, causing misalignment
    - Misaligned comparison reported differences throughout the entire file
    - Result was partial_match with many false positive differences

    After the fix:
    - 60ms offset is correctly detected and applied
    - Encoding differences are filtered (high MFCC correlation)
    - Trailing silence is filtered
    - Result is correctly same_source with no differences
    """
    result = run_compare("source.wav", "source_different_mastering.wav")
    assert result["result"] == "same_source"
    assert result["fingerprint_match"] is True
    # Offset should be detected (approximately -60ms, negative because B starts later)
    assert -0.1 < result["alignment_offset_seconds"] < -0.02
    # No audible differences should be reported
    assert result["differences"] == []
    # Overall similarity should be high (same content)
    assert result["overall_similarity"] > 0.85


# =============================================================================
# 2. Partial Match (Edited) Tests
# =============================================================================


def test_silenced_chunk():
    """2.1 - Detect 0.2s silenced chunk in middle of file."""
    result = run_compare("source_degraded.wav", "edited_silence.wav")
    assert result["result"] == "partial_match"
    assert len(result["differences"]) == 1
    diff = result["differences"][0]
    # Edit is at 18s into degraded audio, plus 3s silence prefix = 21s
    assert 20.5 < diff["start_seconds"] < 21.5
    assert 0.1 < diff["duration_seconds"] < 0.5


def test_reversed_chunk():
    """2.2 - Detect 0.2s reversed chunk in middle of file."""
    result = run_compare("source_degraded.wav", "edited_reversed.wav")
    assert result["result"] == "partial_match"
    assert len(result["differences"]) == 1
    diff = result["differences"][0]
    # Edit is at 18s into degraded audio, plus 3s silence prefix = 21s
    assert 20.5 < diff["start_seconds"] < 21.5
    assert 0.1 < diff["duration_seconds"] < 0.5


def test_chunked_processing_drift():
    """2.3 - Verify accurate boundary detection after 100+ chunks.

    Without the drift fix, frame positions would be off by ~1.5s at this distance.
    """
    result = run_compare("long_source.wav", "long_edited.wav")
    assert result["result"] == "partial_match"
    assert len(result["differences"]) == 1
    diff = result["differences"][0]
    # Edit is at: 3s prefix + 5560.5s repeated prefix + 18s into degraded = 5581.5s
    # Tolerance of 0.5s to account for frame boundaries
    assert 5581.0 < diff["start_seconds"] < 5582.0
    assert 0.1 < diff["duration_seconds"] < 0.5


# =============================================================================
# 2b. Subtle Edit Detection (Phase 2-3 required)
# =============================================================================


def test_phase_inverted_chunk():
    """2.4 - Detect 0.2s phase-inverted chunk.

    Phase inversion (polarity flip) doesn't change magnitude spectrum or energy.
    MFCCs are magnitude-based, so they cannot detect this edit.
    Requires waveform correlation (Phase 3) to detect.
    """
    result = run_compare("source.wav", "edited_phase_inverted.wav")
    assert result["result"] == "partial_match"
    assert len(result["differences"]) == 1
    diff = result["differences"][0]
    assert 17.5 < diff["start_seconds"] < 18.5
    assert 0.1 < diff["duration_seconds"] < 0.5


def test_gain_boosted_chunk():
    """2.5 - Gain differences are filtered as inaudible.

    -12dB = 0.25x amplitude. While detected internally, gain differences
    are filtered out as they represent level adjustments, not content changes.
    Result should be same_source with no reported differences.
    """
    result = run_compare("source.wav", "edited_gain_boost.wav")
    assert result["result"] == "same_source"
    assert result["differences"] == []


def test_lowpass_filtered_chunk():
    """2.6 - Detect 0.2s chunk with low-pass filter (500Hz cutoff).

    Aggressive filtering removes most musical content. Detected via MFCC
    changes (spectral envelope differs significantly) and spectral flux
    (abrupt spectral change at edit boundaries).
    """
    result = run_compare("source.wav", "edited_lowpass.wav")
    assert result["result"] == "partial_match"
    assert len(result["differences"]) == 1
    diff = result["differences"][0]
    assert 17.5 < diff["start_seconds"] < 18.5
    assert 0.1 < diff["duration_seconds"] < 0.5


# =============================================================================
# 3. Different Source Test
# =============================================================================


def test_reversed_audio():
    """3.1 - Fully reversed audio should be classified as completely different.

    This tests temporal feature detection. With delta MFCCs, reversed audio
    has ~0.48 similarity (down from ~0.91 with static MFCCs only) due to
    inverted temporal derivatives. Classification uses fingerprint_match=False
    combined with low similarity to identify as completely_different.
    """
    result = run_compare("source.wav", "reversed.wav")
    assert result["result"] == "completely_different"
    assert (
        result["overall_similarity"] < 0.6
    )  # Delta MFCCs reduce but don't eliminate similarity
    assert result["fingerprint_match"] is False


def test_completely_different():
    """3.2 - White noise should be classified as completely different."""
    result = run_compare("source.wav", "different.wav")
    assert result["result"] == "completely_different"
    assert result["overall_similarity"] < 0.3
    assert result["fingerprint_match"] is False


# =============================================================================
# 4. Long File Reversed Chunk Detection
# =============================================================================


def test_reversed_chunk_long_file():
    """4.1 - Detect 0.2s reversed chunk in long file (100+ chunks).

    This is the critical test for temporal feature detection in chunked processing.
    Without delta MFCCs, reversed chunks become undetectable in long files.
    """
    result = run_compare("long_source.wav", "long_edited_reversed.wav")
    assert result["result"] == "partial_match"
    assert len(result["differences"]) == 1
    diff = result["differences"][0]
    # Edit is at: 3s prefix + 5560.5s repeated prefix + 18s into degraded = 5581.5s
    assert 5581.0 < diff["start_seconds"] < 5582.0
    assert 0.1 < diff["duration_seconds"] < 0.5
