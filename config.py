import os

# Project Base Directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VOICE_MODEL_DIR = os.path.join(MODELS_DIR, "voice_emotion")
TEXT_MODEL_DIR = os.path.join(MODELS_DIR, "text_emotion")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
UI_DIR = os.path.join(PROJECT_ROOT, "ui")

# Multimodal Fusion Configuration
DEFAULT_VOICE_WEIGHT = 0.5
DEFAULT_TEXT_WEIGHT = 0.5

# Shared Standard 8 Emotion Categories across Voice & Text Fusion Space
SHARED_EMOTIONS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
]

# Web Server Settings
SERVER_PORT = 8000
