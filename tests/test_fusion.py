import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fusion.emotion_fusion import fuse_emotions

class TestEmotionFusionModule(unittest.TestCase):
    def setUp(self):
        self.voice_res = {
            "dominant_emotion": "sad",
            "confidence": 0.60,
            "probabilities": {"sad": 0.60, "angry": 0.10, "calm": 0.10, "disgust": 0.05, "fearful": 0.05, "happy": 0.05, "neutral": 0.03, "surprised": 0.02}
        }
        self.text_res = {
            "dominant_emotion": "sad",
            "confidence": 0.80,
            "probabilities": {"sad": 0.80, "angry": 0.05, "calm": 0.02, "disgust": 0.03, "fearful": 0.05, "happy": 0.02, "neutral": 0.02, "surprised": 0.01}
        }

    def test_voice_and_text_fusion(self):
        fused = fuse_emotions(self.voice_res, self.text_res, 0.5, 0.5)
        self.assertEqual(fused["fusion"]["mode"], "voice+text")
        self.assertEqual(fused["final_emotion"], "sad")
        self.assertAlmostEqual(fused["confidence"], 0.70, places=2)

    def test_voice_only_fusion(self):
        fused = fuse_emotions(voice_result=self.voice_res, text_result=None)
        self.assertEqual(fused["fusion"]["mode"], "voice_only")
        self.assertEqual(fused["final_emotion"], "sad")
        self.assertAlmostEqual(fused["confidence"], 0.60, places=2)

    def test_text_only_fusion(self):
        fused = fuse_emotions(voice_result=None, text_result=self.text_res)
        self.assertEqual(fused["fusion"]["mode"], "text_only")
        self.assertEqual(fused["final_emotion"], "sad")
        self.assertAlmostEqual(fused["confidence"], 0.80, places=2)

    def test_fallback_fusion(self):
        fused = fuse_emotions(voice_result=None, text_result=None)
        self.assertEqual(fused["fusion"]["mode"], "fallback_uniform")
        self.assertEqual(fused["final_emotion"], "neutral")
        self.assertEqual(len(fused["probabilities"]), 8)

if __name__ == "__main__":
    unittest.main()
