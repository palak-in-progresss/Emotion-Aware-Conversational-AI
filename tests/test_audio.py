import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.voice_emotion.feature_extractor import extract_audio_features
from models.voice_emotion.feature_extractor_v2 import extract_audio_features_v2
from models.voice_emotion.predict import VoiceEmotionPredictor
from models.voice_emotion.train import parse_ravdess_filename

class TestAudioModule(unittest.TestCase):
    def test_feature_extractor_synthetic_signal(self):
        """Test feature extraction on synthetic sine wave audio signal."""
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        features = extract_audio_features(audio, sr=sr, duration=duration)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (162,))
        self.assertFalse(np.isnan(features).any())

    def test_v2_feature_extractor_synthetic_signal(self):
        """Test V2 feature extraction (342 features) on synthetic sine wave audio signal."""
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        features_v2 = extract_audio_features_v2(audio, sr=sr, duration=duration)
        self.assertIsInstance(features_v2, np.ndarray)
        self.assertEqual(features_v2.shape, (342,))
        self.assertFalse(np.isnan(features_v2).any())

    def test_predictor_fallback_and_inference(self):
        """Test VoiceEmotionPredictor output structure."""
        predictor = VoiceEmotionPredictor()
        sr = 22050
        t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
        audio = 0.3 * np.random.randn(len(t))
        
        result = predictor.predict(audio)
        self.assertIn("dominant_emotion", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn(result["dominant_emotion"], predictor.emotions)
        self.assertEqual(len(result["probabilities"]), 8)

    def test_ravdess_filename_parsing(self):
        """Test RAVDESS filename format parser."""
        filename = "03-01-05-01-01-01-18.wav"
        emotion, actor_id = parse_ravdess_filename(filename)
        self.assertEqual(emotion, "angry")
        self.assertEqual(actor_id, 18)

if __name__ == "__main__":
    unittest.main()
