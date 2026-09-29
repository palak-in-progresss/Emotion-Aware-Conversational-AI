import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from models.voice_emotion.predict import VoiceEmotionPredictor
from models.text_emotion.predict import predict_text_emotion
from fusion.emotion_fusion import fuse_emotions
from audio.speech_to_text import transcribe_audio
from text.response_generator import generate_empathetic_response
from tts.text_to_speech import synthesize_speech
from utils.logger import logger

_voice_predictor = None

def get_voice_predictor():
    global _voice_predictor
    if _voice_predictor is None:
        _voice_predictor = VoiceEmotionPredictor()
    return _voice_predictor

def process_multimodal_input(
    audio_input=None,
    text_input: str = None,
    voice_weight: float = config.DEFAULT_VOICE_WEIGHT,
    text_weight: float = config.DEFAULT_TEXT_WEIGHT
) -> dict:
    """
    End-to-End VIORA Multimodal Pipeline.
    
    1. Transcribes audio if text is missing (STT).
    2. Runs Voice Emotion Recognition (SER).
    3. Runs Text Emotion Recognition (TF-IDF + LogisticRegression).
    4. Fuses probability distributions (Multimodal Probability-Level Fusion).
    5. Generates Empathetic Conversational Response.
    """
    transcript = text_input or ""
    
    # 1. STT if audio provided and no text
    if audio_input and not transcript:
        stt_text = transcribe_audio(audio_input)
        if stt_text:
            transcript = stt_text

    # 2. Voice Emotion Prediction
    voice_res = None
    if audio_input:
        vp = get_voice_predictor()
        voice_res = vp.predict(audio_input)

    # 3. Text Emotion Prediction
    text_res = None
    if transcript and transcript.strip():
        text_res = predict_text_emotion(transcript)

    # 4. Probability-Level Multimodal Fusion
    fused_res = fuse_emotions(
        voice_result=voice_res,
        text_result=text_res,
        voice_weight=voice_weight,
        text_weight=text_weight
    )

    final_emotion = fused_res["final_emotion"]
    confidence = fused_res["confidence"]

    # 5. Empathetic Response Generation
    response_text = generate_empathetic_response(
        user_text=transcript,
        detected_emotion=final_emotion,
        confidence=confidence
    )

    return {
        "transcript": transcript,
        "final_emotion": final_emotion,
        "confidence": confidence,
        "response": response_text,
        "voice": fused_res["voice"],
        "text": fused_res["text"],
        "probabilities": fused_res["probabilities"],
        "fusion": fused_res["fusion"]
    }

def main():
    parser = argparse.ArgumentParser(description="VIORA — Emotion-Aware Conversational AI")
    parser.add_argument("--text", type=str, help="Input text message to analyze")
    parser.add_argument("--audio", type=str, help="Input audio file path (.wav/.mp3) to analyze")
    parser.add_argument("--server", action="store_true", help="Launch the web server HUD UI")
    parser.add_argument("--voice-weight", type=float, default=config.DEFAULT_VOICE_WEIGHT, help="Voice fusion weight (default 0.5)")
    parser.add_argument("--text-weight", type=float, default=config.DEFAULT_TEXT_WEIGHT, help="Text fusion weight (default 0.5)")
    args = parser.parse_args()

    if args.server:
        from server import run_server
        run_server(port=config.SERVER_PORT)
        return

    text_in = args.text
    audio_in = args.audio

    if not text_in and not audio_in:
        print("\n" + "="*70)
        print("          VIORA — MULTIMODAL DEMO PIPELINE EXECUTION           ")
        print("="*70)
        text_in = "I miss you so much and feel really lonely today."
        print(f"Sample Input Text: '{text_in}'")

    res = process_multimodal_input(
        audio_input=audio_in,
        text_input=text_in,
        voice_weight=args.voice_weight,
        text_weight=args.text_weight
    )

    print("\n" + "="*70)
    print("                     VIORA MULTIMODAL RESULT                     ")
    print("="*70)
    print(f"Transcript:        {res['transcript']}")
    print(f"Final Emotion:     {res['final_emotion'].upper()} ({res['confidence']*100:.2f}% confidence)")
    print(f"Fusion Mode:       {res['fusion']['mode']}")
    print(f"Voice Emotion:     {res['voice']['emotion']} (weight: {res['fusion']['voice_weight']})")
    print(f"Text Emotion:      {res['text']['emotion']} (weight: {res['fusion']['text_weight']})")
    print("-" * 70)
    print(f"Empathetic Response:\n{res['response']}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
