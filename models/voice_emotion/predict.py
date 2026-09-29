import os
import json
import joblib
import numpy as np
from models.voice_emotion.feature_extractor_v2 import extract_audio_features_v2
from models.voice_emotion.feature_extractor import extract_audio_features
from utils.logger import logger

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

EXPANDED_MODEL_PATH = os.path.join(MODEL_DIR, "expanded_model.pkl")
EXPANDED_SCALER_PATH = os.path.join(MODEL_DIR, "expanded_scaler.pkl")

CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "champion_model.pkl")
CHAMPION_SCALER_PATH = os.path.join(MODEL_DIR, "champion_scaler.pkl")

V1_MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
V1_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

CONFIG_PATH = os.path.join(MODEL_DIR, "feature_config_v2.json")
V1_CONFIG_PATH = os.path.join(MODEL_DIR, "feature_config.json")

DEFAULT_EMOTIONS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
]

class VoiceEmotionPredictor:
    """
    Predictor for Speech Emotion Recognition (SER).
    Automatically loads the final expanded multi-dataset champion model (342 V2 features)
    or falls back to champion/V1 models if missing.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.emotions = DEFAULT_EMOTIONS
        self.use_v2_features = True
        self.model_version = "None"
        self.is_loaded = False
        
        self.load_model()

    def load_model(self):
        """Loads trained SVM model and scaler, prioritizing the final expanded RAVDESS+CREMA-D model."""
        try:
            # 1. Try loading Final Expanded Model (342 features, RAVDESS + CREMA-D)
            if os.path.exists(EXPANDED_MODEL_PATH) and os.path.exists(EXPANDED_SCALER_PATH):
                self.model = joblib.load(EXPANDED_MODEL_PATH)
                self.scaler = joblib.load(EXPANDED_SCALER_PATH)
                self.use_v2_features = True
                self.model_version = "Final Expanded (RAVDESS + CREMA-D, 342 features)"
                self.is_loaded = True
                logger.info(f"Loaded {self.model_version} from {EXPANDED_MODEL_PATH}")
                return

            # 2. Try loading Champion Model (342 features, RAVDESS only)
            if os.path.exists(CHAMPION_MODEL_PATH) and os.path.exists(CHAMPION_SCALER_PATH):
                self.model = joblib.load(CHAMPION_MODEL_PATH)
                self.scaler = joblib.load(CHAMPION_SCALER_PATH)
                self.use_v2_features = True
                self.model_version = "Champion SVM (RAVDESS, 342 features)"
                self.is_loaded = True
                logger.info(f"Loaded {self.model_version} from {CHAMPION_MODEL_PATH}")
                return

            # 3. Fallback to V1 Model (162 features)
            if os.path.exists(V1_MODEL_PATH) and os.path.exists(V1_SCALER_PATH):
                self.model = joblib.load(V1_MODEL_PATH)
                self.scaler = joblib.load(V1_SCALER_PATH)
                self.use_v2_features = False
                self.model_version = "V1 Baseline (162 features)"
                self.is_loaded = True
                logger.info(f"Loaded {self.model_version} from {V1_MODEL_PATH}")
                return

            logger.warning("No voice emotion model found. Predictor will use acoustic heuristic fallback.")

        except Exception as e:
            logger.error(f"Error loading Voice Emotion model: {e}")

    def predict(self, audio_input):
        """
        Predicts voice emotion probabilities for given audio input (file path or numpy array).
        Returns a dict with 'dominant_emotion', 'confidence', 'probabilities', and 'model_version'.
        """
        # Extract features
        if self.use_v2_features:
            features = extract_audio_features_v2(audio_input)
        else:
            features = extract_audio_features(audio_input)

        if features is None:
            logger.warning("Feature extraction failed. Returning uniform probabilities.")
            return self._fallback_prediction()

        if not self.is_loaded:
            logger.warning("Model not loaded. Returning baseline prediction based on acoustic energy.")
            return self._fallback_acoustic_prediction(features)

        try:
            features_2d = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_2d)

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features_scaled)[0]
            else:
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
                "probabilities": prob_dict,
                "model_version": self.model_version
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
            "probabilities": {e: uniform_prob for e in self.emotions},
            "model_version": "Fallback Uniform"
        }

    def _fallback_acoustic_prediction(self, features):
        """Rule-based acoustic estimate when no model is available."""
        energy = float(np.mean(np.abs(features))) if len(features) > 0 else 0.05
        prob_dict = {e: 0.10 for e in self.emotions}
        if energy > 0.1:
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
            "probabilities": prob_dict,
            "model_version": "Fallback Acoustic Heuristic"
        }
