import os
import sys
import random
from utils.logger import logger

# Non-medical, empathetic response templates per emotion class
EMPATHETIC_FALLBACK_RESPONSES = {
    "sad": [
        "It sounds like you may be feeling sad or going through a tough moment. I'm here to listen whenever you'd like to share more.",
        "I hear that you're feeling down right now. Take all the time you need, and know that your feelings are valid.",
        "It seems like things feel heavy for you at the moment. Remember to be gentle with yourself today."
    ],
    "angry": [
        "It sounds like you're experiencing frustration or anger. It's completely understandable to feel upset when things go wrong.",
        "I hear how frustrating this situation is for you. If you'd like to talk through what happened, I'm here.",
        "It seems like you are dealing with a lot of irritation right now. Taking a slow breath can sometimes help create a little space."
    ],
    "fearful": [
        "It sounds like you might be feeling anxious or worried. You don't have to navigate these feelings alone.",
        "I notice some concern or fear in what you shared. Take a moment to ground yourself—you are safe here.",
        "It seems like you're feeling uneasy or uncertain. Let's take things one small step at a time."
    ],
    "happy": [
        "I'm so glad to hear your positive energy! It sounds like things are going really well for you.",
        "That sounds wonderful! It's great to celebrate these moments of joy.",
        "Your happiness shines through! Thank you for sharing such good news with me."
    ],
    "surprised": [
        "It sounds like you were really caught off guard or surprised! Unexpected moments can certainly take a minute to process.",
        "That definitely sounds like a surprise! How are you feeling now that you've had a moment to absorb it?",
        "Woah, that sounds unexpected! It's completely natural to feel astonished by news like that."
    ],
    "disgust": [
        "It sounds like you're feeling unpleasant or displeased about this situation. That must be uncomfortable.",
        "I can tell this scenario feels off or distasteful to you. It's valid to express your discomfort.",
        "It seems like something left a very negative impression. I'm here if you'd like to discuss it further."
    ],
    "calm": [
        "It sounds like you're feeling peaceful and balanced right now. It's wonderful to experience calm moments.",
        "I appreciate your thoughtful and composed state. How can I assist you in this calm space?",
        "It feels like you are at ease right now. I'm happy to continue our conversation at your pace."
    ],
    "neutral": [
        "I hear you clearly. How are things going with you overall today?",
        "Thank you for sharing that with me. What would you like to explore or talk about next?",
        "I'm listening. Feel free to tell me more about what's on your mind."
    ]
}

class ResponseGenerator:
    """
    Generates empathetic, non-medical conversational responses based on fused user emotion.
    Integrates Google Gemini LLM API if configured, with a rich offline fallback response system.
    """
    def __init__(self):
        self.gemini_available = False
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("PALM_API_KEY")
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.gemini_available = True
                logger.info("Google Gemini API initialized for conversational response generation.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini API: {e}. Active offline response engine.")

    def generate_response(
        self,
        user_text: str = "",
        detected_emotion: str = "neutral",
        confidence: float = 0.5,
        conversation_history: list = None
    ) -> str:
        """
        Generates an empathetic response tailored to user's text and detected emotion.
        Guarantees non-medical, supportive language.
        """
        emotion = (detected_emotion or "neutral").lower()
        if emotion not in EMPATHETIC_FALLBACK_RESPONSES:
            emotion = "neutral"

        if self.gemini_available and user_text.strip():
            try:
                prompt = (
                    "You are VIORA, an empathetic, supportive, emotion-aware conversational AI assistant.\n"
                    f"User Message: '{user_text}'\n"
                    f"Detected User Emotion: {emotion} (Confidence: {confidence*100:.1f}%)\n\n"
                    "Instructions:\n"
                    "1. Respond in 2-3 warm, empathetic, supportive sentences.\n"
                    "2. Acknowledge the user's inferred emotion gently ('It sounds like you may be feeling...').\n"
                    "3. DO NOT make medical or psychological diagnoses.\n"
                    "4. Keep your tone supportive, conversational, and natural."
                )
                response = self.model.generate_content(prompt)
                if response and hasattr(response, "text") and response.text.strip():
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini API generation failed: {e}. Falling back to template generator.")

        # Offline Empathetic Generator Fallback
        templates = EMPATHETIC_FALLBACK_RESPONSES.get(emotion, EMPATHETIC_FALLBACK_RESPONSES["neutral"])
        selected_template = random.choice(templates)
        return selected_template

_response_gen = None

def generate_empathetic_response(
    user_text: str = "",
    detected_emotion: str = "neutral",
    confidence: float = 0.5,
    conversation_history: list = None
) -> str:
    """Public functional interface for response generation."""
    global _response_gen
    if _response_gen is None:
        _response_gen = ResponseGenerator()
    return _response_gen.generate_response(user_text, detected_emotion, confidence, conversation_history)
