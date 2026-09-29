import os
from utils.logger import logger

class TextToSpeech:
    """
    Text-to-Speech audio synthesizer for VIORA responses.
    Uses pyttsx3 or gTTS if available, with graceful fallback handling.
    """
    def __init__(self):
        self.engine_type = None
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine_type = "pyttsx3"
            logger.info("pyttsx3 TTS engine initialized.")
        except Exception as e:
            logger.warning(f"pyttsx3 TTS unavailable: {e}")

    def synthesize(self, text: str, output_path: str = None) -> str:
        """
        Synthesizes text into audio WAV/MP3 file.
        Returns file path if successful, or empty string.
        """
        if not text or not text.strip():
            return ""

        if self.engine_type == "pyttsx3" and output_path:
            try:
                self.engine.save_to_file(text, output_path)
                self.engine.runAndWait()
                return output_path
            except Exception as e:
                logger.warning(f"pyttsx3 synthesis failed: {e}")
                return ""

        return ""

_tts = None

def synthesize_speech(text: str, output_path: str = None) -> str:
    """Public functional interface for TTS synthesis."""
    global _tts
    if _tts is None:
        _tts = TextToSpeech()
    return _tts.synthesize(text, output_path)
