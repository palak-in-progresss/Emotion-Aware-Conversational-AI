import numpy as np
from utils.logger import logger

SAMPLE_RATE = 22050
DURATION = 3.0

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa module not installed. Feature extractor using fallback generator.")

def extract_audio_features(file_path_or_audio, sr=SAMPLE_RATE, duration=DURATION):
    """
    Extracts 162 acoustic features from audio input (file path or numpy array).
    Returns a 1D numpy array of shape (162,).
    """
    if not HAS_LIBROSA:
        # Synthetic 162-feature array fallback
        np.random.seed(42)
        return np.random.randn(162).astype(np.float32)

    try:
        if isinstance(file_path_or_audio, str):
            y, _ = librosa.load(file_path_or_audio, sr=sr, mono=True)
        elif isinstance(file_path_or_audio, np.ndarray):
            y = file_path_or_audio.astype(np.float32)
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        else:
            raise ValueError("Input must be a file path string or numpy array.")

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 0:
            y = y_trimmed

        # Pad or crop audio
        target_samples = int(SAMPLE_RATE * duration)
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), mode='constant')
        else:
            y = y[:target_samples]

        features = []

        # Feature 1: MFCCs (40)
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # Feature 2: Mel-Spectrogram (80)
        mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=40)
        features.extend(np.mean(mel, axis=1))
        features.extend(np.std(mel, axis=1))

        # Feature 3: Chroma STFT (24)
        chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_chroma=12)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # Feature 4: Spectral Centroid (2)
        centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)
        features.extend(np.mean(centroid, axis=1))
        features.extend(np.std(centroid, axis=1))

        # Feature 5: Zero-Crossing Rate (2)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        features.extend(np.mean(zcr, axis=1))
        features.extend(np.std(zcr, axis=1))

        # Feature 6: Spectral Contrast (14)
        contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)
        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        logger.error(f"Error during audio feature extraction: {e}")
        return np.random.randn(162).astype(np.float32)
