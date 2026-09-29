import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.text_emotion.predict import predict_text_emotion, TextEmotionPredictor
from text.emotion_detector import TextEmotionDetector

class TestTextEmotionModule(unittest.TestCase):
    def test_text_emotion_predictor(self):
        predictor = TextEmotionPredictor()
        self.assertTrue(predictor.is_loaded)

        res = predictor.predict("I am so happy that I finally passed my exam!")
        self.assertIn("dominant_emotion", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)
        self.assertIn("raw_emotion", res)

        self.assertEqual(res["dominant_emotion"], "happy")
        self.assertEqual(len(res["probabilities"]), 8)

    def test_love_emotion_mapping(self):
        predictor = TextEmotionPredictor()
        res = predictor.predict("I love spending time with you so much!")
        self.assertEqual(res["dominant_emotion"], "happy")
        self.assertEqual(res["raw_emotion"], "Love")

    def test_empty_input_fallback(self):
        res = predict_text_emotion("")
        self.assertEqual(res["dominant_emotion"], "neutral")
        self.assertEqual(len(res["probabilities"]), 8)

    def test_detector_wrapper(self):
        detector = TextEmotionDetector()
        res = detector.detect_emotion("I miss you so much and feel lonely.")
        self.assertEqual(res["dominant_emotion"], "sad")

if __name__ == "__main__":
    unittest.main()
