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

from models.voice_emotion.feature_extractor import extract_audio_features as extract_v1
from models.voice_emotion.feature_extractor_v2 import extract_audio_features_v2 as extract_v2
from models.voice_emotion.train import parse_ravdess_filename, load_dataset_metadata, DEFAULT_EMOTIONS
from utils.logger import logger

def train_and_eval(df, extractor_fn, version_name, train_actors, test_actors):
    """Evaluates SVM baseline for a given feature extractor on exact real actor split."""
    train_df = df[df["actor_id"].isin(train_actors)].copy()
    test_df = df[df["actor_id"].isin(test_actors)].copy()

    logger.info(f"Extracting features for {version_name} on Train ({len(train_df)}) and Test ({len(test_df)})...")
    X_train = np.array([extractor_fn(fp) for fp in train_df["filepath"]])
    y_train = train_df["emotion"].values

    X_test = np.array([extractor_fn(fp) for fp in test_df["filepath"]])
    y_test = test_df["emotion"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    prec, rec, f1s, supp = precision_recall_fscore_support(y_test, y_pred, labels=DEFAULT_EMOTIONS)
    per_class = {}
    for emo, p, r, f, s in zip(DEFAULT_EMOTIONS, prec, rec, f1s, supp):
        per_class[emo] = {
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f),
            "support": int(s)
        }

    cm = confusion_matrix(y_test, y_pred, labels=DEFAULT_EMOTIONS)
    cm_dict = {emo: list(map(int, row)) for emo, row in zip(DEFAULT_EMOTIONS, cm)}

    return {
        "model": model,
        "scaler": scaler,
        "num_features": X_train.shape[1],
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
            "confusion_matrix": cm_dict
        }
    }

def run_real_v1_vs_v2_experiment(dataset_dir="datasets/real_ravdess", train_actors=list(range(1, 19)), test_actors=list(range(19, 25))):
    """Executes Phase 3 Real RAVDESS V1 vs V2 Feature Engineering Experiment."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info("==================================================")
    logger.info(" PHASE 3: REAL RAVDESS V1 VS V2 FEATURE EXPERIMENT")
    logger.info("==================================================")
    
    df = load_dataset_metadata(dataset_dir)
    if df.empty:
        logger.error(f"No real RAVDESS dataset found at '{dataset_dir}'.")
        return None

    logger.info(f"Dataset samples: {len(df)} | Train Actors: {train_actors} | Test Actors: {test_actors}")

    # Evaluate V1 (162 features)
    res_v1 = train_and_eval(df, extract_v1, "V1 (162 features)", train_actors, test_actors)
    
    # Evaluate V2 (342 features)
    res_v2 = train_and_eval(df, extract_v2, "V2 (342 features)", train_actors, test_actors)

    m1 = res_v1["metrics"]
    m2 = res_v2["metrics"]

    comparison_table = []
    comparison_table.append({
        "Metric": "Macro F1",
        "V1 (162 Feats)": f"{m1['macro_f1']*100:.2f}%",
        "V2 (342 Feats)": f"{m2['macro_f1']*100:.2f}%",
        "Delta": f"{(m2['macro_f1'] - m1['macro_f1'])*100:+.2f}%"
    })
    comparison_table.append({
        "Metric": "Weighted F1",
        "V1 (162 Feats)": f"{m1['weighted_f1']*100:.2f}%",
        "V2 (342 Feats)": f"{m2['weighted_f1']*100:.2f}%",
        "Delta": f"{(m2['weighted_f1'] - m1['weighted_f1'])*100:+.2f}%"
    })
    comparison_table.append({
        "Metric": "Accuracy",
        "V1 (162 Feats)": f"{m1['accuracy']*100:.2f}%",
        "V2 (338 Feats)": f"{m2['accuracy']*100:.2f}%",
        "Delta": f"{(m2['accuracy'] - m1['accuracy'])*100:+.2f}%"
    })
    comparison_table.append({
        "Metric": "Balanced Accuracy",
        "V1 (162 Feats)": f"{m1['balanced_accuracy']*100:.2f}%",
        "V2 (338 Feats)": f"{m2['balanced_accuracy']*100:.2f}%",
        "Delta": f"{(m2['balanced_accuracy'] - m1['balanced_accuracy'])*100:+.2f}%"
    })

    for emo in DEFAULT_EMOTIONS:
        f1_1 = m1["per_class"][emo]["f1_score"]
        f1_2 = m2["per_class"][emo]["f1_score"]
        comparison_table.append({
            "Metric": f"{emo.capitalize()} F1",
            "V1 (162 Feats)": f"{f1_1*100:.2f}%",
            "V2 (342 Feats)": f"{f1_2*100:.2f}%",
            "Delta": f"{(f1_2 - f1_1)*100:+.2f}%"
        })

    comp_df = pd.DataFrame(comparison_table)

    print("\n" + "="*70)
    print("      REAL RAVDESS EXPERIMENTAL RESULTS: V1 vs V2 COMPARISON      ")
    print("="*70)
    print(comp_df.to_string(index=False))
    print("="*70 + "\n")

    # Save V2 Artifacts
    joblib.dump(res_v2["model"], os.path.join(output_dir, "model_v2.pkl"))
    joblib.dump(res_v2["scaler"], os.path.join(output_dir, "scaler_v2.pkl"))

    v2_results = {
        "experiment": "Phase 3: Real RAVDESS V1 vs V2 Feature Engineering",
        "dataset": "REAL RAVDESS (1,440 WAVs)",
        "v1_metrics": m1,
        "v2_metrics": m2,
        "comparison_table": comparison_table
    }
    with open(os.path.join(output_dir, "real_v2_results.json"), "w") as f:
        json.dump(v2_results, f, indent=2)

    logger.info("Saved Real V2 model artifacts: model_v2.pkl, scaler_v2.pkl, real_v2_results.json")
    return comp_df, v2_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3 Real RAVDESS V1 vs V2 Experiment")
    parser.add_argument("--dataset_dir", type=str, default="datasets/real_ravdess", help="Path to real RAVDESS dataset directory")
    args = parser.parse_args()

    run_real_v1_vs_v2_experiment(args.dataset_dir)
