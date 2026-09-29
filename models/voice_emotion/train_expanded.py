import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.voice_emotion.feature_extractor_v2 import extract_audio_features_v2
from models.voice_emotion.train import load_dataset_metadata
from utils.logger import logger

def train_and_eval_expanded_model(
    ravdess_dir="datasets/real_ravdess",
    crema_dir="datasets/crema_d",
    train_actors=list(range(1, 19)),   # RAVDESS Actors 1-18 (1,080 clips)
    test_actors=list(range(19, 25))    # RAVDESS Actors 19-24 (360 clips, UNTOUCHED TEST SET)
):
    """
    Phase 4/6: Trains Champion RBF SVM on RAVDESS (Actors 1-18) + CREMA-D (1,200 clips)
    and evaluates strictly on speaker-independent untouched RAVDESS Test Actors (19-24).
    """
    output_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(output_dir, "expanded_features_cache.joblib")

    logger.info("==================================================")
    logger.info("   PHASE 4/6: MULTI-DATASET EXPANDED MODEL TRAINING")
    logger.info("==================================================")

    # 1. Load RAVDESS metadata
    ravdess_df = load_dataset_metadata(ravdess_dir)
    ravdess_train = ravdess_df[ravdess_df["actor_id"].isin(train_actors)].copy()
    ravdess_test = ravdess_df[ravdess_df["actor_id"].isin(test_actors)].copy()

    # 2. Load CREMA-D metadata
    crema_csv = os.path.join(crema_dir, "metadata.csv")
    if not os.path.exists(crema_csv):
        logger.error(f"CREMA-D metadata not found at {crema_csv}")
        return None

    crema_df = pd.read_csv(crema_csv)
    logger.info(f"Loaded CREMA-D dataset: {len(crema_df)} clips across emotions: {dict(crema_df['emotion'].value_counts())}")

    # Combine training data: RAVDESS Actors 1-18 + CREMA-D
    train_combined_df = pd.concat([
        ravdess_train[["filepath", "emotion"]],
        crema_df[["filepath", "emotion"]]
    ], ignore_index=True)

    logger.info(f"Combined Training set: {len(train_combined_df)} clips (RAVDESS: {len(ravdess_train)}, CREMA-D: {len(crema_df)})")
    logger.info(f"Untouched RAVDESS Test set: {len(ravdess_test)} clips (Actors 19-24)")

    # 3. Extract or load cached features
    if os.path.exists(cache_path):
        logger.info(f"Loading cached expanded feature matrices from {cache_path}")
        cache = joblib.load(cache_path)
        X_train, y_train = cache["X_train"], cache["y_train"]
        X_test, y_test = cache["X_test"], cache["y_test"]
    else:
        logger.info("Extracting V2 342 features for Combined Train set...")
        X_train = np.array([extract_audio_features_v2(fp) for fp in train_combined_df["filepath"]])
        y_train = train_combined_df["emotion"].values

        logger.info("Extracting V2 342 features for Untouched RAVDESS Test set...")
        X_test = np.array([extract_audio_features_v2(fp) for fp in ravdess_test["filepath"]])
        y_test = ravdess_test["emotion"].values

        joblib.dump({"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}, cache_path)

    # Filter out classes that might only exist in test if any (all RAVDESS classes present)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # Filter test clips to only emotions present in training
    known_mask = np.isin(y_test, le.classes_)
    X_test_filtered = X_test[known_mask]
    y_test_filtered = y_test[known_mask]
    y_test_enc = le.transform(y_test_filtered)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_filtered)

    logger.info("Training Champion RBF SVM (C=10.0, gamma=0.001) on RAVDESS + CREMA-D...")
    svm = SVC(kernel='rbf', C=10.0, gamma=0.001, probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train_enc)

    logger.info("Evaluating Expanded Model ONCE on Untouched RAVDESS Test set (Actors 19-24)...")
    pred_test_enc = svm.predict(X_test_scaled)

    macro_f1 = float(f1_score(y_test_enc, pred_test_enc, average="macro"))
    weighted_f1 = float(f1_score(y_test_enc, pred_test_enc, average="weighted"))
    acc = float(accuracy_score(y_test_enc, pred_test_enc))
    bal_acc = float(balanced_accuracy_score(y_test_enc, pred_test_enc))

    prec, rec, f1s, supp = precision_recall_fscore_support(y_test_enc, pred_test_enc, labels=range(len(le.classes_)))
    per_class = {}
    for emo, p, r, f, s in zip(le.classes_, prec, rec, f1s, supp):
        per_class[str(emo)] = {
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f),
            "support": int(s)
        }

    cm = confusion_matrix(y_test_enc, pred_test_enc, labels=range(len(le.classes_)))
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    print("\n" + "="*70)
    print("   MULTI-DATASET EXPANDED MODEL TEST REPORT (RAVDESS + CREMA-D)   ")
    print("="*70)
    print(f"Training Clips Total:         {len(train_combined_df)} (RAVDESS: {len(ravdess_train)}, CREMA-D: {len(crema_df)})")
    print(f"Test Clips (RAVDESS 19-24):   {len(y_test_filtered)}")
    print(f"Expanded Test Accuracy:       {acc*100:.2f}%")
    print(f"Expanded Test Bal Accuracy:   {bal_acc*100:.2f}%")
    print(f"Expanded Test Macro F1-Score: {macro_f1*100:.2f}%")
    print(f"Expanded Test Weighted F1:    {weighted_f1*100:.2f}%")
    print("-" * 70)
    print("Per-Class Classification Report:")
    pred_labels = le.inverse_transform(pred_test_enc)
    print(classification_report(y_test_filtered, pred_labels, target_names=le.classes_, digits=4))
    print("\nConfusion Matrix:")
    print(cm_df.to_string())
    print("="*70 + "\n")

    # Save Expanded Model Artifacts
    joblib.dump(svm, os.path.join(output_dir, "expanded_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "expanded_scaler.pkl"))
    joblib.dump(le, os.path.join(output_dir, "expanded_label_encoder.pkl"))

    results_json = {
        "train_size_total": len(train_combined_df),
        "ravdess_train_size": len(ravdess_train),
        "cremad_train_size": len(crema_df),
        "test_size": len(y_test_filtered),
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class
        }
    }
    with open(os.path.join(output_dir, "expanded_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    logger.info("Saved expanded_model.pkl, expanded_scaler.pkl, and expanded_results.json")
    return results_json

if __name__ == "__main__":
    train_and_eval_expanded_model()
