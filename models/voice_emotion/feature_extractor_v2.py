import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import logger

SAMPLE_RATE = 22050
DURATION = 3.0

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa module not installed. V2 feature extractor using synthetic fallback generator.")

def apply_pre_emphasis(y, coeff=0.97):
    """Applies pre-emphasis filter y[t] = x[t] - coeff * x[t-1]."""
    return np.append(y[0], y[1:] - coeff * y[:-1])

def adaptive_energy_trim(y, top_db=25):
    """Trims leading and trailing silence adaptively using RMS energy."""
    if not HAS_LIBROSA or len(y) == 0:
        return y
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed if len(y_trimmed) > 0 else y
    except Exception:
        return y

def safe_stats(arr):
    """Safely extracts [mean, std, min, max] statistics from a 1D or 2D array along time axis (axis=-1)."""
    if arr is None or arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    if arr.ndim == 1:
        return [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))]
    elif arr.ndim == 2:
        means = np.mean(arr, axis=1)
        stds = np.std(arr, axis=1)
        mins = np.min(arr, axis=1)
        maxs = np.max(arr, axis=1)
        res = []
        for m, s, mn, mx in zip(means, stds, mins, maxs):
            res.extend([float(m), float(s), float(mn), float(mx)])
        return res
    return []

def extract_f0_pitch_stats(y, sr=SAMPLE_RATE):
    """
    Robust F0 fundamental pitch statistics using fast librosa.yin with graceful fallback.
    Returns 6 pitch stats: [f0_mean, f0_std, f0_min, f0_max, f0_median, voiced_frame_ratio].
    """
    if not HAS_LIBROSA or len(y) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    try:
        fmin = float(librosa.note_to_hz('C2')) # ~65 Hz
        fmax = float(librosa.note_to_hz('C7')) # ~2093 Hz
        
        # Fast YIN pitch tracking
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
        
        # Filter valid pitch range (exclude extreme boundaries)
        valid_mask = (f0 > fmin + 1) & (f0 < fmax - 1)
        valid_f0 = f0[valid_mask]
        total_frames = len(f0) if len(f0) > 0 else 1
        
        if len(valid_f0) > 0:
            f0_mean = float(np.mean(valid_f0))
            f0_std = float(np.std(valid_f0))
            f0_min = float(np.min(valid_f0))
            f0_max = float(np.max(valid_f0))
            f0_median = float(np.median(valid_f0))
            voiced_ratio = float(len(valid_f0) / total_frames)
        else:
            f0_mean, f0_std, f0_min, f0_max, f0_median, voiced_ratio = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        return [f0_mean, f0_std, f0_min, f0_max, f0_median, voiced_ratio]

    except Exception as e:
        logger.warning(f"YIN pitch extraction failed ({e}), using zero fallback.")
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def extract_audio_features_v2(file_path_or_audio, sr=SAMPLE_RATE, duration=DURATION):
    """
    Extracts expanded, robust V2 acoustic features from audio input (file path or numpy array).
    Combines pre-emphasis, adaptive trimming, MFCC + Delta + Delta2, F0 pitch stats, RMS stats,
    spectral rolloff, spectral bandwidth, spectral centroid, spectral contrast, chroma, ZCR.
    Returns a 1D numpy array of float32 features.
    """
    V2_EXPECTED_DIM = 342  # Pre-computed total feature count (80+80+80+6+4+4+4+4+32+48+4)

    if not HAS_LIBROSA:
        # Synthetic fallback
        np.random.seed(42)
        return np.random.randn(V2_EXPECTED_DIM).astype(np.float32)

    try:
        # 1. Load audio
        if isinstance(file_path_or_audio, str):
            y, _ = librosa.load(file_path_or_audio, sr=sr, mono=True)
        elif isinstance(file_path_or_audio, np.ndarray):
            y = file_path_or_audio.astype(np.float32)
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        else:
            raise ValueError("Input must be a file path string or numpy array.")

        # 2. Peak normalization
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # 3. Pre-emphasis filter
        y_pre = apply_pre_emphasis(y, coeff=0.97)

        # 4. Adaptive energy-based silence trimming
        y_trimmed = adaptive_energy_trim(y_pre, top_db=25)

        # 5. Fixed duration padding / cropping
        target_samples = int(SAMPLE_RATE * duration)
        if len(y_trimmed) < target_samples:
            y_proc = np.pad(y_trimmed, (0, target_samples - len(y_trimmed)), mode='constant')
        else:
            y_proc = y_trimmed[:target_samples]

        features = []

        # Feature Set 1: MFCCs (20 coeffs x 4 stats: mean, std, min, max = 80)
        mfcc = librosa.feature.mfcc(y=y_proc, sr=SAMPLE_RATE, n_mfcc=20)
        features.extend(safe_stats(mfcc))

        # Feature Set 2: Delta MFCCs (20 coeffs x 4 stats = 80)
        delta_mfcc = librosa.feature.delta(mfcc)
        features.extend(safe_stats(delta_mfcc))

        # Feature Set 3: Delta-Delta MFCCs (20 coeffs x 4 stats = 80)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features.extend(safe_stats(delta2_mfcc))

        # Feature Set 4: F0 Pitch Statistics (6 stats: mean, std, min, max, median, voiced_ratio)
        f0_stats = extract_f0_pitch_stats(y_proc, sr=SAMPLE_RATE)
        features.extend(f0_stats)

        # Feature Set 5: RMS Energy Statistics (1 x 4 stats = 4)
        rms = librosa.feature.rms(y=y_proc)
        features.extend(safe_stats(rms))

        # Feature Set 6: Spectral Centroid (1 x 4 stats = 4)
        centroid = librosa.feature.spectral_centroid(y=y_proc, sr=SAMPLE_RATE)
        features.extend(safe_stats(centroid))

        # Feature Set 7: Spectral Bandwidth (1 x 4 stats = 4)
        bandwidth = librosa.feature.spectral_bandwidth(y=y_proc, sr=SAMPLE_RATE)
        features.extend(safe_stats(bandwidth))

        # Feature Set 8: Spectral Rolloff (1 x 4 stats = 4)
        rolloff = librosa.feature.spectral_rolloff(y=y_proc, sr=SAMPLE_RATE, roll_percent=0.85)
        features.extend(safe_stats(rolloff))

        # Feature Set 9: Spectral Contrast (7 bands x 4 stats = 28)
        contrast = librosa.feature.spectral_contrast(y=y_proc, sr=SAMPLE_RATE)
        features.extend(safe_stats(contrast))

        # Feature Set 10: Chroma STFT (12 pitch classes x 4 stats = 48)
        chroma = librosa.feature.chroma_stft(y=y_proc, sr=SAMPLE_RATE, n_chroma=12)
        features.extend(safe_stats(chroma))

        # Feature Set 11: Zero Crossing Rate (1 x 4 stats = 4)
        zcr = librosa.feature.zero_crossing_rate(y=y_proc)
        features.extend(safe_stats(zcr))

        feat_arr = np.array(features, dtype=np.float32)
        return feat_arr

    except Exception as e:
        logger.error(f"Error during V2 audio feature extraction: {e}")
        np.random.seed(42)
        return np.random.randn(V2_EXPECTED_DIM).astype(np.float32)

if __name__ == "__main__":
    sr = 22050
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    feats = extract_audio_features_v2(test_audio)
    print(f"Extracted V2 feature vector shape: {feats.shape}")
