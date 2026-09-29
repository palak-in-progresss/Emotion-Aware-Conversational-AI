import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
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

from models.voice_emotion.feature_extractor_v2 import extract_audio_features_v2
from models.voice_emotion.train import load_dataset_metadata, DEFAULT_EMOTIONS
from utils.logger import logger

def benchmark_model_candidates(
    dataset_dir="datasets/real_ravdess",
    train_actors=list(range(1, 15)),   # Actors 1-14 (14 actors = 840 clips)
    val_actors=list(range(15, 19)),    # Actors 15-18 (4 actors = 240 clips)
    test_actors=list(range(19, 25))    # Actors 19-24 (6 actors = 360 clips, UNTOUCHED)
):
    """
    Phase 5: Benchmarks 5 model architectures on real RAVDESS V2 features (342 dims)
    using TRAIN vs VALIDATION actors for model selection without test set leakage.
    """
    output_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(output_dir, "v2_features_cache.joblib")

    logger.info("==================================================")
    logger.info("   PHASE 5: REAL RAVDESS MODEL ARCHITECTURE BENCHMARK")
    logger.info("==================================================")

    df = load_dataset_metadata(dataset_dir)
    if df.empty:
        logger.error(f"No dataset found at '{dataset_dir}'.")
        return None

    train_df = df[df["actor_id"].isin(train_actors)].copy()
    val_df = df[df["actor_id"].isin(val_actors)].copy()
    test_df = df[df["actor_id"].isin(test_actors)].copy()

    logger.info(f"Train set: {len(train_df)} clips | Val set: {len(val_df)} clips | Final Test set: {len(test_df)} clips")

    if os.path.exists(cache_path):
        logger.info(f"Loading pre-computed V2 feature matrices from cache: {cache_path}")
        cache = joblib.load(cache_path)
        X_train, y_train = cache["X_train"], cache["y_train"]
        X_val, y_val = cache["X_val"], cache["y_val"]
        X_test, y_test = cache["X_test"], cache["y_test"]
    else:
        logger.info("Extracting V2 342 features for Train, Val, and Test splits...")
        X_train = np.array([extract_audio_features_v2(fp) for fp in train_df["filepath"]])
        y_train = train_df["emotion"].values

        X_val = np.array([extract_audio_features_v2(fp) for fp in val_df["filepath"]])
        y_val = val_df["emotion"].values

        X_test = np.array([extract_audio_features_v2(fp) for fp in test_df["filepath"]])
        y_test = test_df["emotion"].values

        joblib.dump({"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test}, cache_path)

    # Encode labels to integers for MLP compatibility
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    candidates = {}

    # 1. RBF SVM Grid Search
    logger.info("Tuning RBF SVM Hyperparameters on Validation set...")
    best_svm = None
    best_svm_val_f1 = -1.0
    for C in [0.1, 1.0, 10.0, 50.0, 100.0]:
        for gamma in ['scale', 'auto', 0.001, 0.01, 0.1]:
            svm = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
            svm.fit(X_train_scaled, y_train_enc)
            pred_val = svm.predict(X_val_scaled)
            val_f1 = f1_score(y_val_enc, pred_val, average='macro')
            if val_f1 > best_svm_val_f1:
                best_svm_val_f1 = val_f1
                best_svm = (f"RBF SVM (C={C}, gamma={gamma})", svm)

    candidates["RBF_SVM"] = best_svm

    # 2. ExtraTrees Classifier
    logger.info("Training ExtraTrees Classifier...")
    et = ExtraTreesClassifier(n_estimators=300, max_depth=16, min_samples_split=3, random_state=42)
    et.fit(X_train_scaled, y_train_enc)
    candidates["ExtraTrees"] = ("ExtraTrees (300 trees)", et)

    # 3. Random Forest Classifier
    logger.info("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_split=3, random_state=42)
    rf.fit(X_train_scaled, y_train_enc)
    candidates["RandomForest"] = ("RandomForest (300 trees)", rf)

    # 4. HistGradientBoosting Classifier (Fast XGBoost equivalent)
    logger.info("Training HistGradientBoosting Classifier...")
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)
    hgb.fit(X_train_scaled, y_train_enc)
    candidates["HistGradientBoosting"] = ("HistGradientBoosting (lr=0.05)", hgb)

    # 5. MLP Neural Network Classifier
    logger.info("Training Multi-Layer Perceptron (MLP) Neural Network...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400, early_stopping=True, n_iter_no_change=15, random_state=42)
    mlp.fit(X_train_scaled, y_train_enc)
    candidates["MLP_NeuralNet"] = ("MLP Neural Net (256x128)", mlp)

    # Evaluate candidates on VALIDATION set for selection
    val_results = []
    champion_key = None
    champion_name = None
    champion_model = None
    champion_val_f1 = -1.0

    for key, (name, model) in candidates.items():
        val_pred = model.predict(X_val_scaled)
        macro_f1 = float(f1_score(y_val_enc, val_pred, average="macro"))
        acc = float(accuracy_score(y_val_enc, val_pred))
        bal_acc = float(balanced_accuracy_score(y_val_enc, val_pred))
        weighted_f1 = float(f1_score(y_val_enc, val_pred, average="weighted"))

        val_results.append({
            "Architecture": name,
            "Val Macro F1": f"{macro_f1*100:.2f}%",
            "Val Weighted F1": f"{weighted_f1*100:.2f}%",
            "Val Accuracy": f"{acc*100:.2f}%",
            "Val Bal Accuracy": f"{bal_acc*100:.2f}%"
        })

        if macro_f1 > champion_val_f1:
            champion_val_f1 = macro_f1
            champion_key = key
            champion_name = name
            champion_model = model

    val_summary_df = pd.DataFrame(val_results)

    print("\n" + "="*70)
    print("      MODEL SELECTION SUMMARY (EVALUATED ON VALIDATION ACTORS 15-18)     ")
    print("="*70)
    print(val_summary_df.to_string(index=False))
    print(f"\n[CHAMPION] ARCHITECTURE SELECTED BY VALIDATION MACRO F1: {champion_name} ({champion_val_f1*100:.2f}%)")
    print("="*70 + "\n")

    # Final UNTOUCHED Test Evaluation for Champion Model
    logger.info(f"Evaluating Champion Model ({champion_name}) ONCE on Untouched Test Actors (19-24)...")
    
    # Re-fit champion on combined Train + Val (Actors 1-18) for maximum data before test eval
    X_trainval = np.vstack([X_train, X_val])
    y_trainval_enc = np.hstack([y_train_enc, y_val_enc])
    scaler_final = StandardScaler()
    X_trainval_scaled = scaler_final.fit_transform(X_trainval)
    X_test_scaled_final = scaler_final.transform(X_test)

    champion_model.fit(X_trainval_scaled, y_trainval_enc)
    final_test_pred_enc = champion_model.predict(X_test_scaled_final)

    final_macro_f1 = float(f1_score(y_test_enc, final_test_pred_enc, average="macro"))
    final_weighted_f1 = float(f1_score(y_test_enc, final_test_pred_enc, average="weighted"))
    final_acc = float(accuracy_score(y_test_enc, final_test_pred_enc))
    final_bal_acc = float(balanced_accuracy_score(y_test_enc, final_test_pred_enc))

    prec, rec, f1s, supp = precision_recall_fscore_support(y_test_enc, final_test_pred_enc, labels=range(len(le.classes_)))
    per_class_final = {}
    for emo, p, r, f, s in zip(le.classes_, prec, rec, f1s, supp):
        per_class_final[str(emo)] = {
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f),
            "support": int(s)
        }

    cm_final = confusion_matrix(y_test_enc, final_test_pred_enc, labels=range(len(le.classes_)))
    cm_final_df = pd.DataFrame(cm_final, index=le.classes_, columns=le.classes_)

    print("\n" + "="*70)
    print("   FINAL UNTOUCHED TEST EVALUATION REPORT (CHAMPION MODEL ON ACTORS 19-24) ")
    print("="*70)
    print(f"Champion Architecture:        {champion_name}")
    print(f"Final Test Accuracy:          {final_acc*100:.2f}%")
    print(f"Final Test Balanced Accuracy: {final_bal_acc*100:.2f}%")
    print(f"Final Test Macro F1-Score:    {final_macro_f1*100:.2f}%")
    print(f"Final Test Weighted F1-Score: {final_weighted_f1*100:.2f}%")
    print("-" * 70)
    print("Per-Class Classification Report:")
    final_pred_labels = le.inverse_transform(final_test_pred_enc)
    print(classification_report(y_test, final_pred_labels, target_names=le.classes_, digits=4))
    print("\nConfusion Matrix:")
    print(cm_final_df.to_string())
    print("="*70 + "\n")

    # Save Champion Model Artifacts
    joblib.dump(champion_model, os.path.join(output_dir, "champion_model.pkl"))
    joblib.dump(scaler_final, os.path.join(output_dir, "champion_scaler.pkl"))
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

    benchmark_json = {
        "validation_summary": val_results,
        "champion_architecture": champion_name,
        "final_test_metrics": {
            "accuracy": final_acc,
            "balanced_accuracy": final_bal_acc,
            "macro_f1": final_macro_f1,
            "weighted_f1": final_weighted_f1,
            "per_class": per_class_final
        }
    }
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(benchmark_json, f, indent=2)

    logger.info("Saved champion_model.pkl, champion_scaler.pkl, label_encoder.pkl, and benchmark_results.json")
    return val_summary_df, benchmark_json

if __name__ == "__main__":
    benchmark_model_candidates()
