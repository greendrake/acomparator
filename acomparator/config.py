"""Configuration dataclass for audio comparison."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration parameters for audio comparison."""

    same_threshold: float
    different_threshold: float
    min_diff_duration: float
    diff_threshold: float
    output_dir: Path
    sample_rate: int = 22050
    n_mfcc: int = 13
    hop_length: int = 512
    use_fingerprint: bool = True
