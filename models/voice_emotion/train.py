import os
import sys
import json
import glob
import re
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.voice_emotion.feature_extractor import extract_audio_features
from utils.logger import logger

EMOTION_MAP_RAVDESS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

DEFAULT_EMOTIONS = sorted(list(EMOTION_MAP_RAVDESS.values()))

def parse_ravdess_filename(filepath):
    """
    Parses RAVDESS audio filename format: 03-01-EMOTION-INTENSITY-STATEMENT-REPEAT-ACTOR.wav
    Returns (emotion_name, actor_id_int) or (None, None) if parsing fails.
    """
    filename = os.path.basename(filepath)
    parts = filename.replace(".wav", "").split("-")
    if len(parts) == 7:
        emotion_code = parts[2]
        actor_code = parts[6]
        emotion = EMOTION_MAP_RAVDESS.get(emotion_code)
        try:
            actor_id = int(actor_code)
            return emotion, actor_id
        except ValueError:
            pass
    return None, None

def load_dataset_metadata(dataset_dir):
    """
    Scans dataset directory recursively for wav files and parses emotion and actor ID metadata.
    """
    wav_files = glob.glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True)
    records = []
    
    for filepath in wav_files:
        emotion, actor_id = parse_ravdess_filename(filepath)
        if emotion is None:
            # Try folder-based fallback (e.g. Actor_01/happy.wav or happy/01.wav)
            folder_name = os.path.basename(os.path.dirname(filepath)).lower()
            for emo in DEFAULT_EMOTIONS:
                if emo in folder_name:
                    emotion = emo
                    break
            # Try extracting actor ID from folder
            actor_match = re.search(r"actor_?(\d+)", filepath, re.IGNORECASE)
            if actor_match:
                actor_id = int(actor_match.group(1))

        if emotion and actor_id:
            records.append({
                "filepath": filepath,
                "emotion": emotion,
                "actor_id": actor_id
            })
            
    df = pd.DataFrame(records)
    return df

def train_ser_baseline(dataset_dir, train_actors=list(range(1, 19)), test_actors=list(range(19, 25)), output_dir=None):
    """
    Executes reproducible Actor-Independent Speech Emotion Recognition (SER) training pipeline.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    logger.info("==================================================")
    logger.info("  VIORA SER Baseline Training Pipeline (Step 1)   ")
    logger.info("==================================================")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Training Actors: {train_actors}")
    logger.info(f"Test Actors (Unseen Speakers): {test_actors}")

    df = load_dataset_metadata(dataset_dir)
    if df.empty:
        logger.error(f"No valid RAVDESS audio files found in '{dataset_dir}'. Please provide RAVDESS audio files.")
        return None

    logger.info(f"Found {len(df)} total audio samples across {df['actor_id'].nunique()} actors.")

    # Split dataset by actor IDs
    train_df = df[df["actor_id"].isin(train_actors)].copy()
    test_df = df[df["actor_id"].isin(test_actors)].copy()

    if train_df.empty or test_df.empty:
        logger.error(f"Split results in empty partition! Train: {len(train_df)}, Test: {len(test_df)}")
        return None

    logger.info(f"Train split: {len(train_df)} samples | Test split: {len(test_df)} samples")

    # Class distributions
    train_dist = train_df["emotion"].value_counts().to_dict()
    test_dist = test_df["emotion"].value_counts().to_dict()
    logger.info(f"Train Class Distribution: {train_dist}")
    logger.info(f"Test Class Distribution:  {test_dist}")

    # Extract acoustic features
    logger.info("Extracting 162 acoustic features for Training split...")
    X_train = np.array([extract_audio_features(fp) for fp in train_df["filepath"]])
    y_train = train_df["emotion"].values

    logger.info("Extracting 162 acoustic features for Test split (Unseen speakers)...")
    X_test = np.array([extract_audio_features(fp) for fp in test_df["filepath"]])
    y_test = test_df["emotion"].values

    # Fit StandardScaler strictly on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM Model (RBF Kernel baseline)
    logger.info("Training SVM Classifier (RBF Kernel)...")
    model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate on Unseen Speakers
    logger.info("Evaluating on Unseen Test Speakers...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # Calculate Metrics
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    prec, rec, f1s, supp = precision_recall_fscore_support(y_test, y_pred, labels=DEFAULT_EMOTIONS)
    per_class_metrics = {}
    for emo, p, r, f, s in zip(DEFAULT_EMOTIONS, prec, rec, f1s, supp):
        per_class_metrics[emo] = {
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f),
            "support": int(s)
        }

    cm = confusion_matrix(y_test, y_pred, labels=DEFAULT_EMOTIONS)
    cm_dict = {emo: list(map(int, row)) for emo, row in zip(DEFAULT_EMOTIONS, cm)}

    results = {
        "model_version": "v1.0-baseline",
        "actor_split": {
            "train_actors": list(map(int, train_actors)),
            "test_actors": list(map(int, test_actors))
        },
        "samples": {
            "train_count": len(train_df),
            "test_count": len(test_df)
        },
        "class_distributions": {
            "train": train_dist,
            "test": test_dist
        },
        "overall_metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm_dict,
        "emotions_order": DEFAULT_EMOTIONS
    }

    # Print Report
    print("\n" + "="*60)
    print("           VIORA SER BASELINE EVALUATION REPORT           ")
    print("="*60)
    print(f"Accuracy:          {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1*100:.2f}%")
    print(f"Weighted F1-Score: {weighted_f1*100:.2f}%")
    print("-"*60)
    print("Per-Class Performance breakdown:")
    print(classification_report(y_test, y_pred, target_names=DEFAULT_EMOTIONS, digits=4))

    # Save Model Artifacts
    model_path = os.path.join(output_dir, "model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    config_path = os.path.join(output_dir, "feature_config.json")
    results_path = os.path.join(output_dir, "baseline_results.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    config_data = {
        "emotions": DEFAULT_EMOTIONS,
        "num_features": X_train.shape[1],
        "model_type": "SVC-RBF",
        "sample_rate": 22050,
        "duration": 3.0
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved trained model to:   {model_path}")
    logger.info(f"Saved scaler to:          {scaler_path}")
    logger.info(f"Saved feature config to:  {config_path}")
    logger.info(f"Saved baseline JSON to:   {results_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VIORA Speech Emotion Recognition (SER) Baseline")
    parser.add_argument("--dataset_dir", type=str, default="datasets/audio", help="Path to audio dataset directory")
    parser.add_argument("--train_actors", type=int, nargs="+", default=list(range(1, 19)), help="Actor IDs for training")
    parser.add_argument("--test_actors", type=int, nargs="+", default=list(range(19, 25)), help="Actor IDs for testing")
    args = parser.parse_args()

    train_ser_baseline(args.dataset_dir, train_actors=args.train_actors, test_actors=args.test_actors)
