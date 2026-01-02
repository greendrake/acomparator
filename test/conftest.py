"""Test fixtures for acomparator test suite."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, filtfilt

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SOURCE_FILE = FIXTURES_DIR / "source.mp3"
OUTPUT_DIR = Path("output")

GENERATED_FILES = [
    "source.wav",
    "source_copy.wav",
    "source_silence_prefix.wav",
    "source_degraded.wav",
    "source_with_trailing.wav",
    "source_encoding_then_content_diff.wav",
    "source_different_mastering.wav",
    "source_fadein_offset.wav",
    "edited_silence.wav",
    "edited_reversed.wav",
    "edited_phase_inverted.wav",
    "edited_gain_boost.wav",
    "edited_lowpass.wav",
    "long_source.wav",
    "long_edited.wav",
    "long_edited_reversed.wav",
    "reversed.wav",
    "different.wav",
]


@pytest.fixture(scope="session", autouse=True)
def generate_fixtures():
    """Generate test fixtures before tests, clean up after."""
    source, sr = _load_audio_ffmpeg(SOURCE_FILE)
    _generate_all_fixtures(source, sr)

    yield

    for filename in GENERATED_FILES:
        filepath = FIXTURES_DIR / filename
        if filepath.exists():
            filepath.unlink()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def _load_audio_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Load audio file using ffmpeg, returning (samples, sample_rate)."""
    import io

    # Convert to WAV via pipe
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    audio, sr = sf.read(io.BytesIO(result.stdout))
    return audio, sr


def _generate_all_fixtures(source: np.ndarray, sr: int) -> None:
    silence_3s = np.zeros(int(3 * sr))

    # 1.1 - Two identical WAV files
    sf.write(FIXTURES_DIR / "source.wav", source, sr)
    sf.write(FIXTURES_DIR / "source_copy.wav", source, sr)

    # 1.2 - Silence prefix (3s)
    sf.write(
        FIXTURES_DIR / "source_silence_prefix.wav",
        np.concatenate([silence_3s, source]),
        sr,
    )

    # 1.3 - Degraded: -3dB volume, low-level noise, 3s silence offset
    degraded = source * _db_to_linear(-3)
    degraded = degraded + _generate_noise(len(source), -30)
    sf.write(
        FIXTURES_DIR / "source_degraded.wav", np.concatenate([silence_3s, degraded]), sr
    )

    # 1.4 - Source with trailing content (extra 5s at end)
    # Append first 5 seconds of source again to create different-length file
    trailing_content = source[: int(5 * sr)]
    sf.write(
        FIXTURES_DIR / "source_with_trailing.wav",
        np.concatenate([source, trailing_content]),
        sr,
    )

    # 1.5 - Encoding difference region followed by content difference
    # This tests that encoding differences are filtered and only content
    # differences are reported. Structure:
    # - 0-10s: identical
    # - 10-20s: encoding difference (add subtle noise - same spectrum, different waveform)
    # - 20-25s: actual content difference (different audio)
    # - 25s onwards: identical
    enc_diff_start = int(10 * sr)
    enc_diff_end = int(20 * sr)
    content_diff_start = int(20 * sr)
    content_diff_end = int(25 * sr)

    # Create encoding difference: add small uncorrelated noise
    # Noise level is low enough to not affect MFCC but enough to reduce waveform correlation
    # 5% amplitude ≈ -26dB
    enc_noise = _generate_noise(enc_diff_end - enc_diff_start, -26)

    # Create content difference: use reversed audio (clearly different content)
    different_content = source[content_diff_start:content_diff_end][::-1]

    encoding_then_content = np.concatenate(
        [
            source[:enc_diff_start],  # 0-10s: identical
            source[enc_diff_start:enc_diff_end] + enc_noise,  # 10-20s: encoding diff
            different_content,  # 20-25s: content diff
            source[content_diff_end:],  # 25s+: identical
        ]
    )
    sf.write(
        FIXTURES_DIR / "source_encoding_then_content_diff.wav",
        encoding_then_content,
        sr,
    )

    # 1.6 - Different mastering with offset and trailing silence
    # Simulates comparing two different rips/masters of the same source:
    # - 60ms start offset (one starts slightly earlier)
    # - Subtle noise throughout (different encoding artifacts)
    # - 2s trailing silence
    # Should be detected as same_source with no audible differences
    offset_60ms = np.zeros(int(0.060 * sr))  # 60ms silence prefix
    trailing_2s = np.zeros(int(2.0 * sr))  # 2s trailing silence

    # Add subtle encoding noise (similar to different codec artifacts)
    # Use very low level noise that affects waveform but not perceptual content
    different_mastering = source + _generate_noise(len(source), -40)
    # Clip to valid range
    different_mastering = np.clip(different_mastering, -1.0, 1.0)

    sf.write(
        FIXTURES_DIR / "source_different_mastering.wav",
        np.concatenate([offset_60ms, different_mastering, trailing_2s]),
        sr,
    )

    # 1.7 - Fade-in alignment test: refinement validation prevents bad alignment
    # This tests that waveform refinement validation prevents incorrect alignments
    # when fade-in timing could mislead the cross-correlation.
    #
    # Structure:
    # - File A (source.wav): Original audio
    # - File B (source_fadein_offset.wav):
    #   - 2s silence prefix
    #   - 100ms of very low-level "pre-fade" noise (simulates early fade-in start)
    #   - 200ms fade-in with the noise blended into actual audio
    #   - Rest of audio identical to source (offset by ~2.1s total)
    #
    # The pre-fade noise creates a scenario where:
    # - Coarse alignment (chroma) gives ~-2.1s offset
    # - Old refinement might lock onto the pre-fade noise timing
    # - New refinement validates correlation and keeps the correct offset
    silence_2s = np.zeros(int(2.0 * sr))

    # Create pre-fade noise (very low level, represents early fade-in start)
    prefade_len = int(0.1 * sr)  # 100ms
    prefade_noise = np.random.randn(prefade_len) * 0.001  # Very quiet noise

    # Apply fade-in to first 200ms of audio
    fadein_len = int(0.2 * sr)
    fadein_curve = np.linspace(0, 1, fadein_len) ** 2  # Quadratic fade-in
    source_with_fadein = source.copy()
    source_with_fadein[:fadein_len] = source[:fadein_len] * fadein_curve

    sf.write(
        FIXTURES_DIR / "source_fadein_offset.wav",
        np.concatenate([silence_2s, prefade_noise, source_with_fadein]),
        sr,
    )

    # 2.1 - Silenced chunk at 18s (0.2s duration)
    edit_pos = int(18 * sr)
    edit_len = int(0.2 * sr)
    edited_silence = _apply_edit(degraded, edit_pos, edit_len, np.zeros(edit_len))
    sf.write(
        FIXTURES_DIR / "edited_silence.wav",
        np.concatenate([silence_3s, edited_silence]),
        sr,
    )

    # 2.2 - Reversed chunk at 18s
    reversed_chunk = degraded[edit_pos : edit_pos + edit_len][::-1]
    edited_reversed = _apply_edit(degraded, edit_pos, edit_len, reversed_chunk)
    sf.write(
        FIXTURES_DIR / "edited_reversed.wav",
        np.concatenate([silence_3s, edited_reversed]),
        sr,
    )

    # 2.4 - Phase-inverted chunk at 18s (requires waveform correlation to detect)
    inverted_chunk = -source[edit_pos : edit_pos + edit_len]
    edited_phase = _apply_edit(source, edit_pos, edit_len, inverted_chunk)
    sf.write(FIXTURES_DIR / "edited_phase_inverted.wav", edited_phase, sr)

    # 2.5 - Gain change (-12dB) at 18s (requires amplitude detection)
    # -12dB = 0.25x amplitude - significant but doesn't clip
    reduced_chunk = source[edit_pos : edit_pos + edit_len] * _db_to_linear(-12)
    edited_gain = _apply_edit(source, edit_pos, edit_len, reduced_chunk)
    sf.write(FIXTURES_DIR / "edited_gain_boost.wav", edited_gain, sr)

    # 2.6 - Low-pass filtered chunk at 18s (requires spectral flux to detect)
    # 500Hz cutoff removes most musical content, making it clearly detectable
    filtered_chunk = _lowpass_filter(source[edit_pos : edit_pos + edit_len], 500, sr)
    edited_lowpass = _apply_edit(source, edit_pos, edit_len, filtered_chunk)
    sf.write(FIXTURES_DIR / "edited_lowpass.wav", edited_lowpass, sr)

    # 2.3 - Long files for chunked processing drift test
    # ~5500s prefix (100 chunks) to reproduce accumulated drift scenario
    # Write in chunks to avoid memory issues
    _write_long_file(
        FIXTURES_DIR / "long_source.wav", silence_3s, source, degraded, sr, 149
    )
    _write_long_file(
        FIXTURES_DIR / "long_edited.wav", silence_3s, source, edited_silence, sr, 149
    )
    _write_long_file(
        FIXTURES_DIR / "long_edited_reversed.wav",
        silence_3s,
        source,
        edited_reversed,
        sr,
        149,
    )

    # 3.1 - Fully reversed source (tests temporal feature detection)
    sf.write(FIXTURES_DIR / "reversed.wav", source[::-1], sr)

    # 3.2 - White noise (completely different content)
    different = _generate_noise(len(source), 0)  # 0dB = full scale
    sf.write(FIXTURES_DIR / "different.wav", different, sr)


def _db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude ratio."""
    return 10 ** (db / 20)


def _generate_noise(length: int, db_level: float) -> np.ndarray:
    """Generate normalized white noise at a specific dB level."""
    noise = np.random.randn(length)
    noise = noise / np.max(np.abs(noise))  # Normalize to [-1, 1]
    return noise * _db_to_linear(db_level)


def _apply_edit(
    source: np.ndarray,
    edit_pos: int,
    edit_len: int,
    modified_chunk: np.ndarray,
) -> np.ndarray:
    """Apply an edit to source audio by replacing a chunk at edit_pos."""
    return np.concatenate(
        [source[:edit_pos], modified_chunk, source[edit_pos + edit_len :]]
    )


def _lowpass_filter(audio: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    """Apply a low-pass filter to audio."""
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    b, a = butter(4, normalized_cutoff, btype="low")
    return filtfilt(b, a, audio)


def _write_long_file(
    path: Path,
    silence: np.ndarray,
    source: np.ndarray,
    suffix: np.ndarray,
    sr: int,
    repeat_count: int,
) -> None:
    """Write a long audio file by streaming chunks to avoid memory issues."""
    with sf.SoundFile(path, mode="w", samplerate=sr, channels=1) as f:
        f.write(silence)
        for _ in range(repeat_count):
            f.write(source)
        f.write(suffix)
