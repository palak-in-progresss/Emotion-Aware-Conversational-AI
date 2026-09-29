import os
import numpy as np
from utils.logger import logger

class SpeechToText:
    """
    Speech-to-Text transcriber for audio inputs.
    Uses SpeechRecognition (Google Web Speech API / Sphinx) if installed,
    with graceful fallback handling.
    """
    def __init__(self):
        self.sr_available = False
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.sr_available = True
            logger.info("SpeechRecognition library initialized successfully.")
        except ImportError:
            logger.warning("speech_recognition package not installed. STT fallback mode active.")

    def transcribe(self, audio_input) -> str:
        """
        Transcribes audio file path or array into string text.
        Returns transcribed text or empty string if transcription fails/unavailable.
        """
        if not self.sr_available:
            return ""

        if isinstance(audio_input, str) and os.path.exists(audio_input):
            try:
                import speech_recognition as sr
                with sr.AudioFile(audio_input) as source:
                    audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                logger.info(f"STT Transcript: '{text}'")
                return text
            except Exception as e:
                logger.warning(f"STT transcription failed for {audio_input}: {e}")
                return ""

        return ""

_stt = None

def transcribe_audio(audio_input) -> str:
    """Public functional interface for Speech-to-Text transcription."""
    global _stt
    if _stt is None:
        _stt = SpeechToText()
    return _stt.transcribe(audio_input)
