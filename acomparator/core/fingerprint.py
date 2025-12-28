"""Chromaprint audio fingerprinting for quick similarity check."""

from pathlib import Path

import acoustid
import chromaprint


class FingerprintError(Exception):
    """Raised when fingerprinting fails."""


def compute_fingerprint(path: Path) -> tuple[int, list[int]]:
    """
    Compute chromaprint fingerprint for audio file.

    Args:
        path: Path to audio file.

    Returns:
        Tuple of (duration in seconds, fingerprint as list of integers).

    Raises:
        FingerprintError: If fingerprinting fails.
    """
    try:
        duration, fp_encoded = acoustid.fingerprint_file(str(path))
        fp_decoded, _version = chromaprint.decode_fingerprint(fp_encoded)
        if fp_decoded is None:
            raise FingerprintError(f"Failed to decode fingerprint for {path}")
        return duration, list(fp_decoded)
    except Exception as e:
        raise FingerprintError(f"Failed to compute fingerprint for {path}: {e}") from e


def compare_fingerprints(fp_a: list[int], fp_b: list[int]) -> float:
    """
    Compare two fingerprints using bit error rate.

    Args:
        fp_a: First fingerprint as list of integers.
        fp_b: Second fingerprint as list of integers.

    Returns:
        Similarity score in range [0, 1], where 1 means identical.
    """
    if not fp_a or not fp_b:
        return 0.0

    # Compare overlapping portion
    min_len = min(len(fp_a), len(fp_b))
    if min_len == 0:
        return 0.0

    # Count matching bits
    total_bits = 0
    matching_bits = 0

    for i in range(min_len):
        xor = fp_a[i] ^ fp_b[i]
        # Count differing bits
        differing = bin(xor & 0xFFFFFFFF).count("1")
        total_bits += 32
        matching_bits += 32 - differing

    # Penalize length difference
    max_len = max(len(fp_a), len(fp_b))
    length_penalty = min_len / max_len

    return (matching_bits / total_bits) * length_penalty
