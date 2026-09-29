from models.text_emotion.predict import predict_text_emotion, TextEmotionPredictor

class TextEmotionDetector:
    """Wrapper class for text emotion detection in the VIORA pipeline."""
    def __init__(self):
        self.predictor = TextEmotionPredictor()

    def detect_emotion(self, text: str) -> dict:
        return self.predictor.predict(text)

__all__ = ["predict_text_emotion", "TextEmotionPredictor", "TextEmotionDetector"]
