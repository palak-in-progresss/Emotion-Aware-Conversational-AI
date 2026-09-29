import os
import json
import pickle
import numpy as np
from utils.logger import logger

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_model.pkl")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# Shared 8 VIORA emotion classes
SHARED_EMOTIONS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
]

# Direct mapping from 8 raw text categories to 8 shared VIORA SER categories
# Note: 'Love' maps to 'happy' (positive valence) with documented handling rule,
# while preserving 'Love' in raw_emotion metadata.
TEXT_TO_SHARED_MAP = {
    "Joy": "happy",
    "Sadness": "sad",
    "Anger": "angry",
    "Fear": "fearful",
    "Surprise": "surprised",
    "Love": "happy",
    "Disgust": "disgust",
    "Neutral": "neutral"
}

class TextEmotionPredictor:
    """
    Predictor for Text Emotion Recognition.
    Loads TF-IDF + LogisticRegression pipeline and maps 8-class text predictions
    into VIORA's standardized shared 8-class emotion probability space.
    """
    def __init__(self, model_path=MODEL_PATH, label_map_path=LABEL_MAP_PATH):
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.pipeline = None
        self.raw_classes = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Love", "Disgust", "Neutral"]
        self.is_loaded = False
        
        self.load_model()

    def load_model(self):
        """Loads saved TF-IDF pipeline and label map."""
        try:
            if os.path.exists(self.label_map_path):
                with open(self.label_map_path, "r") as f:
                    label_map = json.load(f)
                    self.raw_classes = sorted(label_map.keys(), key=lambda k: label_map[k])

            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    self.pipeline = pickle.load(f)
                self.is_loaded = True
                logger.info(f"TextEmotionPredictor loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Text emotion model not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading Text Emotion model: {e}")

    def predict(self, text: str) -> dict:
        """
        Predicts text emotion for a given raw string.
        Returns standardized 8-class VIORA probability distribution and raw emotion metadata.
        """
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty or invalid text input provided. Returning fallback prediction.")
            return self._fallback_prediction()

        if not self.is_loaded:
            logger.warning("Text emotion model not loaded. Returning heuristic fallback prediction.")
            return self._heuristic_fallback(text)

        try:
            raw_probs = self.pipeline.predict_proba([text])[0]
            raw_prob_dict = {cls: float(p) for cls, p in zip(self.raw_classes, raw_probs)}
            
            raw_dominant_idx = int(np.argmax(raw_probs))
            raw_dominant_emotion = self.raw_classes[raw_dominant_idx]

            # Map raw text probabilities into 8 shared SER emotion categories
            shared_probs = {e: 0.001 for e in SHARED_EMOTIONS}  # small smoothing epsilon
            for raw_cls, p in raw_prob_dict.items():
                target_cls = TEXT_TO_SHARED_MAP.get(raw_cls, "neutral")
                if target_cls in shared_probs:
                    shared_probs[target_cls] += p

            # Re-normalize shared probability distribution
            total = sum(shared_probs.values())
            shared_probs = {e: float(p / total) for e, p in shared_probs.items()}

            dominant_emotion = max(shared_probs, key=shared_probs.get)
            confidence = float(shared_probs[dominant_emotion])

            return {
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "probabilities": shared_probs,
                "raw_emotion": raw_dominant_emotion,
                "raw_probabilities": raw_prob_dict
            }

        except Exception as e:
            logger.error(f"Error during text emotion prediction: {e}")
            return self._fallback_prediction()

    def _fallback_prediction(self) -> dict:
        """Uniform probability distribution fallback."""
        uniform_p = 1.0 / len(SHARED_EMOTIONS)
        return {
            "dominant_emotion": "neutral",
            "confidence": uniform_p,
            "probabilities": {e: uniform_p for e in SHARED_EMOTIONS},
            "raw_emotion": "Neutral"
        }

    def _heuristic_fallback(self, text: str) -> dict:
        """Simple keyword-based fallback if model file is missing."""
        text_lower = text.lower()
        shared_probs = {e: 0.05 for e in SHARED_EMOTIONS}

        if any(w in text_lower for w in ["happy", "great", "awesome", "love", "joy", "good"]):
            shared_probs["happy"] = 0.60
            dom = "happy"
            raw = "Joy"
        elif any(w in text_lower for w in ["sad", "miss", "sorry", "cry", "lonely", "hurt"]):
            shared_probs["sad"] = 0.60
            dom = "sad"
            raw = "Sadness"
        elif any(w in text_lower for w in ["angry", "hate", "mad", "outrageous", "furious"]):
            shared_probs["angry"] = 0.60
            dom = "angry"
            raw = "Anger"
        else:
            shared_probs["neutral"] = 0.60
            dom = "neutral"
            raw = "Neutral"

        total = sum(shared_probs.values())
        shared_probs = {e: float(p / total) for e, p in shared_probs.items()}

        return {
            "dominant_emotion": dom,
            "confidence": shared_probs[dom],
            "probabilities": shared_probs,
            "raw_emotion": raw
        }

_text_predictor = None

def predict_text_emotion(text: str) -> dict:
    """Public functional interface for text emotion prediction."""
    global _text_predictor
    if _text_predictor is None:
        _text_predictor = TextEmotionPredictor()
    return _text_predictor.predict(text)
