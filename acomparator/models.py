"""Data models for audio comparison results."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ComparisonResult(Enum):
    """Result classification for audio comparison."""

    SAME_SOURCE = "same_source"
    COMPLETELY_DIFFERENT = "completely_different"
    PARTIAL_MATCH = "partial_match"


@dataclass
class DifferenceSection:
    """A section where the two audio files differ."""

    start_seconds: float
    end_seconds: float
    duration_seconds: float
    file_a_segment: Path | None  # None if section missing from file A
    file_b_segment: Path | None  # None if section missing from file B
    comment: str | None = None  # Explains subtle/inaudible differences


@dataclass
class ComparisonReport:
    """Complete report from audio comparison."""

    result: ComparisonResult
    overall_similarity: float
    alignment_offset_seconds: float
    fingerprint_match: bool | None  # None if fingerprint check was skipped
    differences: list[DifferenceSection]
