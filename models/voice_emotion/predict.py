import os
import json
import joblib
import numpy as np
from models.voice_emotion.feature_extractor import extract_audio_features
from utils.logger import logger

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "feature_config.json")

DEFAULT_EMOTIONS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
]

class VoiceEmotionPredictor:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH, config_path=CONFIG_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        self.model = None
        self.scaler = None
        self.emotions = DEFAULT_EMOTIONS
        self.is_loaded = False
        
        self.load_model()

    def load_model(self):
        """Loads trained SVM model, StandardScaler, and feature configuration metadata."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    self.emotions = config.get("emotions", DEFAULT_EMOTIONS)
                    logger.info(f"Loaded feature_config.json with {len(self.emotions)} emotions.")

            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_loaded = True
                logger.info(f"VoiceEmotionPredictor loaded model successfully from {self.model_path}")
            else:
                logger.warning(
                    f"Model or Scaler not found at {self.model_path}. "
                    "Place model.pkl, scaler.pkl, and feature_config.json in models/voice_emotion/"
                )
        except Exception as e:
            logger.error(f"Error loading Voice Emotion model: {e}")

    def predict(self, audio_input):
        """
        Predicts voice emotion probabilities for given audio input (file path or numpy array).
        Returns a dict with 'dominant_emotion', 'confidence', and 'probabilities'.
        """
        # Extract features
        features = extract_audio_features(audio_input)
        if features is None:
            logger.warning("Feature extraction failed. Returning uniform probabilities.")
            return self._fallback_prediction()

        if not self.is_loaded:
            logger.warning("Model not loaded. Returning baseline prediction based on acoustic energy.")
            return self._fallback_acoustic_prediction(features)

        try:
            # Reshape for single sample and scale
            features_2d = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_2d)

            # Predict probabilities
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features_scaled)[0]
            else:
                # Decision function fallback
                df_scores = self.model.decision_function(features_scaled)[0]
                exp_scores = np.exp(df_scores - np.max(df_scores))
                probs = exp_scores / exp_scores.sum()

            prob_dict = {emotion: float(prob) for emotion, prob in zip(self.emotions, probs)}
            dominant_idx = int(np.argmax(probs))
            dominant_emotion = self.emotions[dominant_idx]
            confidence = float(probs[dominant_idx])

            return {
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "probabilities": prob_dict
            }

        except Exception as e:
            logger.error(f"Error during voice emotion prediction: {e}")
            return self._fallback_prediction()

    def _fallback_prediction(self):
        """Uniform probability fallback if prediction fails."""
        uniform_prob = 1.0 / len(self.emotions)
        return {
            "dominant_emotion": "neutral",
            "confidence": uniform_prob,
            "probabilities": {e: uniform_prob for e in self.emotions}
        }

    def _fallback_acoustic_prediction(self, features):
        """Rule-based acoustic estimate when model.pkl is not yet present."""
        # Spectral centroid is around index 144
        zcr_mean = float(features[146]) if len(features) > 146 else 0.05
        
        prob_dict = {e: 0.10 for e in self.emotions}
        if zcr_mean > 0.15:
            prob_dict["surprised"] = 0.40
            prob_dict["angry"] = 0.20
            dominant = "surprised"
        else:
            prob_dict["calm"] = 0.40
            prob_dict["neutral"] = 0.30
            dominant = "calm"
            
        total = sum(prob_dict.values())
        prob_dict = {k: v / total for k, v in prob_dict.items()}
        
        return {
            "dominant_emotion": dominant,
            "confidence": prob_dict[dominant],
            "probabilities": prob_dict
        }
