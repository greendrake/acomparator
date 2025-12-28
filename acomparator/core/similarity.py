"""Similarity algorithms for audio comparison."""

from typing import Protocol

import librosa
import numpy as np
from numpy.linalg import norm


class SimilarityAlgorithm(Protocol):
    """Protocol for similarity computation algorithms."""

    def compute_frame_similarity(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        sample_rate: int,
        hop_length: int,
    ) -> np.ndarray:
        """
        Compute per-frame similarity between two audio signals.

        Args:
            audio_a: First audio signal.
            audio_b: Second audio signal.
            sample_rate: Sample rate of both signals.
            hop_length: Hop length for frame extraction.

        Returns:
            Array of similarity scores in range [0, 1] for each frame.
        """
        ...


class MFCCCosineSimilarity:
    """MFCC extraction with cosine similarity per frame, enhanced with energy comparison.

    Uses static MFCCs plus delta (first derivative) and delta-delta (second derivative)
    coefficients to capture temporal dynamics. This enables detection of reversed audio
    and subtle temporal edits that static MFCCs alone would miss.
    """

    def __init__(self, n_mfcc: int = 13):
        self.n_mfcc = n_mfcc

    def compute_frame_similarity(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        sample_rate: int,
        hop_length: int,
    ) -> np.ndarray:
        """
        Compute per-frame similarity using MFCCs with temporal derivatives.

        Combines:
        - Static MFCCs (13 coefficients) - spectral envelope
        - Delta MFCCs (13 coefficients) - rate of change
        - Delta-delta MFCCs (13 coefficients) - acceleration

        The delta features capture temporal dynamics, making reversed audio
        detectable (delta values invert when audio is reversed).

        Also includes energy ratio comparison to detect silence vs audio.

        Args:
            audio_a: First audio signal.
            audio_b: Second audio signal.
            sample_rate: Sample rate of both signals.
            hop_length: Hop length for MFCC extraction.

        Returns:
            Array of similarity scores in range [0, 1] for each frame.
        """
        # Extract MFCCs with delta and delta-delta
        features_a = self._extract_features(audio_a, sample_rate, hop_length)
        features_b = self._extract_features(audio_b, sample_rate, hop_length)

        # Compute RMS energy per frame
        rms_a = librosa.feature.rms(y=audio_a, hop_length=hop_length)[0]
        rms_b = librosa.feature.rms(y=audio_b, hop_length=hop_length)[0]

        # Ensure same number of frames by padding shorter arrays
        n_features = features_a.shape[
            0
        ]  # 39 features (13 static + 13 delta + 13 delta-delta)
        n_frames_a = features_a.shape[1]
        n_frames_b = features_b.shape[1]
        n_frames = max(n_frames_a, n_frames_b)

        if n_frames_a < n_frames:
            padding = np.zeros((n_features, n_frames - n_frames_a))
            features_a = np.hstack([features_a, padding])
        if n_frames_b < n_frames:
            padding = np.zeros((n_features, n_frames - n_frames_b))
            features_b = np.hstack([features_b, padding])

        # Pad RMS arrays
        if len(rms_a) < n_frames:
            rms_a = np.pad(rms_a, (0, n_frames - len(rms_a)))
        if len(rms_b) < n_frames:
            rms_b = np.pad(rms_b, (0, n_frames - len(rms_b)))

        # Compute cosine similarity per frame
        mfcc_similarity = np.zeros(n_frames)
        for i in range(n_frames):
            vec_a = features_a[:, i]
            vec_b = features_b[:, i]
            norm_a = norm(vec_a)
            norm_b = norm(vec_b)

            if norm_a > 0 and norm_b > 0:
                cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
                # Cosine similarity is in [-1, 1], normalize to [0, 1]
                mfcc_similarity[i] = (cos_sim + 1) / 2
            else:
                mfcc_similarity[i] = 0.0

        # Compute energy similarity for amplitude change detection
        # Penalize significant energy differences that suggest edits
        # Don't penalize minor loudness differences from mastering (~3dB = 0.7 ratio)
        energy_similarity = np.ones(n_frames)
        for i in range(n_frames):
            e_a = rms_a[i]
            e_b = rms_b[i]
            max_e = max(e_a, e_b)
            min_e = min(e_a, e_b)
            # Only consider it a difference if:
            # 1. The louder signal has significant energy (> 0.01 after normalization)
            # 2. The ratio is significant (< 0.4, i.e., > 8dB difference)
            # This detects -12dB edits (0.25 ratio) but ignores -3dB mastering (0.7 ratio)
            if max_e > 0.01 and min_e / max_e < 0.4:
                energy_similarity[i] = min_e / max_e

        # Phase 3: Waveform correlation per frame
        # Detects phase inversion and subtle amplitude changes
        waveform_similarity = self._compute_waveform_correlation(
            audio_a, audio_b, hop_length, n_frames
        )

        # Phase 2: Spectral flux comparison
        # Detects abrupt spectral changes (EQ, filtering)
        flux_similarity = self._compute_spectral_flux_similarity(
            audio_a, audio_b, sample_rate, hop_length, n_frames
        )

        # Spectral centroid comparison
        # Detects EQ/filtering that shifts the spectral center
        centroid_similarity = self._compute_centroid_similarity(
            audio_a, audio_b, sample_rate, hop_length, n_frames
        )

        # Combine all similarity measures using minimum
        # Any detection method can flag a difference
        similarity = np.minimum(mfcc_similarity, energy_similarity)
        similarity = np.minimum(similarity, waveform_similarity)
        similarity = np.minimum(similarity, flux_similarity)
        similarity = np.minimum(similarity, centroid_similarity)

        return similarity

    def _extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_length: int,
    ) -> np.ndarray:
        """
        Extract MFCC features with delta and delta-delta coefficients.

        Each feature group is normalized to unit variance so they contribute
        equally to cosine similarity. Without normalization, static MFCCs
        (magnitude ~30) would dominate over deltas (magnitude ~2.5).

        Args:
            audio: Audio signal.
            sample_rate: Sample rate.
            hop_length: Hop length for feature extraction.

        Returns:
            Feature matrix of shape (39, n_frames) containing normalized:
            - Rows 0-12: Static MFCCs
            - Rows 13-25: Delta MFCCs (first derivative)
            - Rows 26-38: Delta-delta MFCCs (second derivative)
        """
        # Extract static MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=self.n_mfcc, hop_length=hop_length
        )

        # Compute delta (first derivative) and delta-delta (second derivative)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Normalize each feature group to unit variance
        # This ensures deltas contribute equally despite smaller magnitude
        mfcc_norm = self._normalize_features(mfcc)
        delta_norm = self._normalize_features(mfcc_delta)
        delta2_norm = self._normalize_features(mfcc_delta2)

        # Stack all features: static + delta + delta-delta
        features = np.vstack([mfcc_norm, delta_norm, delta2_norm])

        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to zero mean and unit variance per coefficient.

        Args:
            features: Feature matrix of shape (n_coefficients, n_frames).

        Returns:
            Normalized feature matrix.
        """
        # Compute mean and std per coefficient (across frames)
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)

        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)

        return (features - mean) / std

    def _compute_waveform_correlation(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        hop_length: int,
        n_frames: int,
    ) -> np.ndarray:
        """
        Compute per-frame waveform correlation (Phase 3).

        Compares actual sample values within each frame. Detects:
        - Phase inversion (correlation ≈ -1)
        - Sample-level edits (correlation drops significantly)

        Only flags frames where correlation indicates a fundamental change
        (phase inversion or uncorrelated content), not minor differences
        from mastering or compression.

        Args:
            audio_a: First audio signal.
            audio_b: Second audio signal.
            hop_length: Frame hop length in samples.
            n_frames: Number of frames.

        Returns:
            Array of similarity scores in range [0, 1] for each frame.
        """
        similarity = np.ones(n_frames)

        for i in range(n_frames):
            start = i * hop_length
            end = start + hop_length

            # Extract frame samples
            frame_a = (
                audio_a[start:end] if end <= len(audio_a) else np.zeros(hop_length)
            )
            frame_b = (
                audio_b[start:end] if end <= len(audio_b) else np.zeros(hop_length)
            )

            # Handle edge cases
            if len(frame_a) < hop_length:
                frame_a = np.pad(frame_a, (0, hop_length - len(frame_a)))
            if len(frame_b) < hop_length:
                frame_b = np.pad(frame_b, (0, hop_length - len(frame_b)))

            # Compute normalized cross-correlation at zero lag
            norm_a = np.sqrt(np.sum(frame_a**2))
            norm_b = np.sqrt(np.sum(frame_b**2))

            if norm_a > 1e-8 and norm_b > 1e-8:
                correlation = np.dot(frame_a, frame_b) / (norm_a * norm_b)
                # Only flag as different if correlation strongly indicates
                # phase inversion (correlation < -0.9). This strict threshold
                # avoids false positives from:
                # - Alignment jitter (few ms offset can cause moderate negative correlation)
                # - Slight waveform differences from encoding
                # - Comparing silence vs audio due to alignment error
                # True phase inversion gives correlation ≈ -1.0
                if correlation < -0.9:
                    # Map correlation [-1, -0.9] to similarity [0, 0.05]
                    similarity[i] = (correlation + 1) / 2
                # else: leave at 1.0
            else:
                # Silent or near-silent frames are considered similar
                similarity[i] = 1.0

        return similarity

    def _compute_spectral_flux_similarity(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        sample_rate: int,
        hop_length: int,
        n_frames: int,
    ) -> np.ndarray:
        """
        Compute spectral flux similarity (Phase 2).

        Compares onset strength envelopes to detect abrupt spectral changes.
        Effective for detecting EQ changes, filtering, and other spectral edits.

        Only flags significant differences (> 50% relative change) to avoid
        false positives from normal mastering variations.

        Args:
            audio_a: First audio signal.
            audio_b: Second audio signal.
            sample_rate: Sample rate.
            hop_length: Frame hop length.
            n_frames: Number of frames.

        Returns:
            Array of similarity scores in range [0, 1] for each frame.
        """
        # Compute onset strength (spectral flux)
        onset_a = librosa.onset.onset_strength(
            y=audio_a, sr=sample_rate, hop_length=hop_length
        )
        onset_b = librosa.onset.onset_strength(
            y=audio_b, sr=sample_rate, hop_length=hop_length
        )

        # Pad to same length
        if len(onset_a) < n_frames:
            onset_a = np.pad(onset_a, (0, n_frames - len(onset_a)))
        if len(onset_b) < n_frames:
            onset_b = np.pad(onset_b, (0, n_frames - len(onset_b)))

        onset_a = onset_a[:n_frames]
        onset_b = onset_b[:n_frames]

        # Compute similarity based on relative difference
        # Only flag very large differences (> 50% relative change)
        max_onset = np.maximum(onset_a, onset_b)
        diff = np.abs(onset_a - onset_b)
        relative_diff = diff / (max_onset + 1e-8)

        similarity = np.ones(n_frames)
        # Only penalize where:
        # 1. There's significant onset activity (max > 0.5)
        # 2. The relative difference is large (> 50%)
        significant_mask = (max_onset > 0.5) & (relative_diff > 0.5)
        similarity[significant_mask] = 1.0 - np.minimum(
            relative_diff[significant_mask], 1.0
        )

        return similarity

    def _compute_centroid_similarity(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        sample_rate: int,
        hop_length: int,
        n_frames: int,
    ) -> np.ndarray:
        """
        Compute spectral centroid similarity.

        The spectral centroid is the "center of mass" of the spectrum.
        Lowpass filtering dramatically shifts the centroid downward.
        This detects EQ/filtering edits that MFCCs might miss because
        MFCCs capture spectral envelope shape, not absolute position.

        Only flags large centroid differences (> 50% shift) to avoid
        false positives from normal audio variations.

        Args:
            audio_a: First audio signal.
            audio_b: Second audio signal.
            sample_rate: Sample rate.
            hop_length: Frame hop length.
            n_frames: Number of frames.

        Returns:
            Array of similarity scores in range [0, 1] for each frame.
        """
        # Compute spectral centroid per frame
        centroid_a = librosa.feature.spectral_centroid(
            y=audio_a, sr=sample_rate, hop_length=hop_length
        )[0]
        centroid_b = librosa.feature.spectral_centroid(
            y=audio_b, sr=sample_rate, hop_length=hop_length
        )[0]

        # Pad to same length
        if len(centroid_a) < n_frames:
            centroid_a = np.pad(centroid_a, (0, n_frames - len(centroid_a)))
        if len(centroid_b) < n_frames:
            centroid_b = np.pad(centroid_b, (0, n_frames - len(centroid_b)))

        centroid_a = centroid_a[:n_frames]
        centroid_b = centroid_b[:n_frames]

        # Compute similarity based on centroid ratio
        # A 500Hz lowpass on 2800Hz content gives ratio ~0.18
        max_centroid = np.maximum(centroid_a, centroid_b)
        min_centroid = np.minimum(centroid_a, centroid_b)

        similarity = np.ones(n_frames)
        for i in range(n_frames):
            if (
                max_centroid[i] > 500
            ):  # Only consider frames with meaningful spectral content
                ratio = min_centroid[i] / max_centroid[i]
                # Flag if centroid shifts by more than 50% (ratio < 0.5)
                if ratio < 0.5:
                    similarity[i] = ratio
            # else: leave at 1.0 (low-frequency or silent content)

        return similarity
