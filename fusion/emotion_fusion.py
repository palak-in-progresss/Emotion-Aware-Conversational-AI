import os
import sys
import numpy as np
from utils.logger import logger

SHARED_EMOTIONS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
]

def fuse_emotions(
    voice_result: dict = None,
    text_result: dict = None,
    voice_weight: float = 0.5,
    text_weight: float = 0.5
) -> dict:
    """
    Performs probability-level multimodal emotion fusion between voice and text predictions.

    Supports:
      - Voice + Text (both modalities present)
      - Voice only (when text is missing/empty)
      - Text only (when voice is missing/empty)

    Parameters
    ----------
    voice_result : dict, optional
        Output from VoiceEmotionPredictor (must contain 'probabilities')
    text_result : dict, optional
        Output from TextEmotionPredictor (must contain 'probabilities')
    voice_weight : float, default 0.5
        Reliability weight assigned to voice modality
    text_weight : float, default 0.5
        Reliability weight assigned to text modality

    Returns
    -------
    dict
        Explainable multimodal fusion result including final emotion, confidence,
        normalized probability vector, modality details, and fusion weights.
    """
    has_voice = bool(voice_result and isinstance(voice_result, dict) and "probabilities" in voice_result)
    has_text = bool(text_result and isinstance(text_result, dict) and "probabilities" in text_result)

    # Determine active fusion mode
    if has_voice and has_text:
        mode = "voice+text"
        w_v = max(0.0, float(voice_weight))
        w_t = max(0.0, float(text_weight))
        total_w = w_v + w_t
        if total_w <= 0:
            w_v, w_t, total_w = 0.5, 0.5, 1.0
        w_v /= total_w
        w_t /= total_w
    elif has_voice:
        mode = "voice_only"
        w_v, w_t = 1.0, 0.0
    elif has_text:
        mode = "text_only"
        w_v, w_t = 0.0, 1.0
    else:
        mode = "fallback_uniform"
        w_v, w_t = 0.0, 0.0

    fused_probs = {}

    if mode == "fallback_uniform":
        uniform_p = 1.0 / len(SHARED_EMOTIONS)
        fused_probs = {e: uniform_p for e in SHARED_EMOTIONS}
    else:
        v_probs = voice_result["probabilities"] if has_voice else {}
        t_probs = text_result["probabilities"] if has_text else {}

        for emotion in SHARED_EMOTIONS:
            pv = float(v_probs.get(emotion, 0.0))
            pt = float(t_probs.get(emotion, 0.0))
            fused_probs[emotion] = w_v * pv + w_t * pt

        # Re-normalize probability vector
        sum_p = sum(fused_probs.values())
        if sum_p > 0:
            fused_probs = {e: float(p / sum_p) for e, p in fused_probs.items()}
        else:
            uniform_p = 1.0 / len(SHARED_EMOTIONS)
            fused_probs = {e: uniform_p for e in SHARED_EMOTIONS}

    if mode == "fallback_uniform":
        dominant_emotion = "neutral"
    else:
        dominant_emotion = max(fused_probs, key=fused_probs.get)
    confidence = float(fused_probs[dominant_emotion])

    voice_summary = {
        "emotion": voice_result.get("dominant_emotion", "N/A") if has_voice else "N/A",
        "confidence": float(voice_result.get("confidence", 0.0)) if has_voice else 0.0,
        "model_version": voice_result.get("model_version", "N/A") if has_voice else "N/A"
    }

    text_summary = {
        "emotion": text_result.get("dominant_emotion", "N/A") if has_text else "N/A",
        "confidence": float(text_result.get("confidence", 0.0)) if has_text else 0.0,
        "raw_emotion": text_result.get("raw_emotion", "N/A") if has_text else "N/A"
    }

    fusion_summary = {
        "voice_weight": w_v,
        "text_weight": w_t,
        "mode": mode
    }

    return {
        "final_emotion": dominant_emotion,
        "confidence": confidence,
        "probabilities": fused_probs,
        "voice": voice_summary,
        "text": text_summary,
        "fusion": fusion_summary
    }
