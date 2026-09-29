import os
import sys
import json
import glob
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.voice_emotion.feature_extractor import extract_audio_features
from models.voice_emotion.train import parse_ravdess_filename, load_dataset_metadata, DEFAULT_EMOTIONS

def audit_v1_baseline(dataset_dir="datasets/real_ravdess", model_path="models/voice_emotion/model.pkl", scaler_path="models/voice_emotion/scaler.pkl"):
    """
    Audits and evaluates pre-trained model.pkl and scaler.pkl on real RAVDESS test actors (Actors 19-24).
    """
    print("\n==================================================")
    print("      VIORA SER BASELINE AUDIT ON REAL RAVDESS    ")
    print("==================================================\n")

    model_abs = os.path.abspath(model_path)
    scaler_abs = os.path.abspath(scaler_path)
    
    print(f"1. Model file loaded:  {model_abs} (Exists: {os.path.exists(model_abs)})")
    print(f"2. Scaler file loaded: {scaler_abs} (Exists: {os.path.exists(scaler_abs)})")

    if not os.path.exists(model_abs) or not os.path.exists(scaler_abs):
        print("ERROR: Pre-trained model.pkl or scaler.pkl not found!")
        return

    model = joblib.load(model_abs)
    scaler = joblib.load(scaler_abs)

    df = load_dataset_metadata(dataset_dir)
    print(f"\n3. Total files found in '{dataset_dir}': {len(df)}")
    
    train_actors = list(range(1, 19))
    test_actors = list(range(19, 25))
    
    print(f"   - Train Actor IDs: {train_actors}")
    print(f"   - Test Actor IDs:  {test_actors}")

    train_df = df[df["actor_id"].isin(train_actors)].copy()
    test_df = df[df["actor_id"].isin(test_actors)].copy()

    print(f"   - Number of Train Clips: {len(train_df)}")
    print(f"   - Number of Test Clips:  {len(test_df)}")

    train_dist = train_df["emotion"].value_counts().to_dict() if not train_df.empty else {}
    test_dist = test_df["emotion"].value_counts().to_dict() if not test_df.empty else {}

    print(f"\n4. Train Class Distribution: {train_dist}")
    print(f"5. Test Class Distribution:  {test_dist}")

    if test_df.empty:
        print("ERROR: Test dataframe is empty! Cannot evaluate.")
        return

    print("\n6. Extracting 162 V1 acoustic features for held-out Test set (360 real audio clips)...")
    X_test_raw = np.array([extract_audio_features(fp) for fp in test_df["filepath"]])
    y_test = test_df["emotion"].values

    print(f"   - Feature matrix shape: {X_test_raw.shape} (Dimensionality: {X_test_raw.shape[1]})")

    X_test_scaled = scaler.transform(X_test_raw)

    y_pred_raw = model.predict(X_test_scaled)
    
    classes = list(model.classes_)
    if isinstance(classes[0], (int, np.integer)):
        y_pred = [DEFAULT_EMOTIONS[i] if i < len(DEFAULT_EMOTIONS) else str(i) for i in y_pred_raw]
    else:
        y_pred = list(map(str, y_pred_raw))

    y_test_str = list(map(str, y_test))

    macro_f1 = float(f1_score(y_test_str, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test_str, y_pred, average="weighted"))
    acc = float(accuracy_score(y_test_str, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test_str, y_pred))
    cm = confusion_matrix(y_test_str, y_pred, labels=DEFAULT_EMOTIONS)

    print("\n==================================================")
    print("  REAL RAVDESS HELD-OUT TEST EVALUATION REPORT    ")
    print("==================================================")
    print(f"Macro F1 Score:    {macro_f1*100:.2f}%")
    print(f"Weighted F1 Score: {weighted_f1*100:.2f}%")
    print(f"Accuracy:          {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print("\nPer-Class Classification Report:")
    print(classification_report(y_test_str, y_pred, target_names=DEFAULT_EMOTIONS, digits=4))
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=DEFAULT_EMOTIONS, columns=DEFAULT_EMOTIONS)
    print(cm_df.to_string())
    print("==================================================\n")

    audit_res = {
        "dataset": "REAL RAVDESS",
        "train_clips": len(train_df),
        "test_clips": len(test_df),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "balanced_accuracy": bal_acc
    }
    with open("models/voice_emotion/real_v1_baseline.json", "w") as f:
        json.dump(audit_res, f, indent=2)

if __name__ == "__main__":
    audit_v1_baseline()
