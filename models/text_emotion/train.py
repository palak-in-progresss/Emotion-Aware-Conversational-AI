import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger import logger

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_DATA_DIR = os.path.join(PROJECT_ROOT, "scratch", "text_emotion_zip", "emotion detection through text", "data", "processed")

RAW_TEXT_LABELS = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Love", "Disgust", "Neutral"]

def train_text_emotion_model():
    """
    Trains the TF-IDF + LogisticRegression text emotion classifier on the combined
    dair-ai/emotion + GoEmotions dataset (61,627 samples) and saves artifacts to models/text_emotion/.
    """
    logger.info("==================================================")
    logger.info("       TRAINING TEXT EMOTION CLASSIFIER           ")
    logger.info("==================================================")

    train_csv = os.path.join(ZIP_DATA_DIR, "train.csv")
    val_csv = os.path.join(ZIP_DATA_DIR, "val.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        logger.error(f"Text dataset CSVs not found in {ZIP_DATA_DIR}")
        return None

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    X_train, y_train = train_df["text"].astype(str).tolist(), train_df["label"].tolist()
    X_val, y_val = val_df["text"].astype(str).tolist(), val_df["label"].tolist()

    logger.info(f"Loaded Text Data -> Train: {len(X_train):,}, Val: {len(X_val):,}")

    classes = np.arange(len(RAW_TEXT_LABELS))
    weights = compute_class_weight("balanced", classes=classes, y=np.array(y_train))
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, max_iter=1000, class_weight=class_weight_dict, solver="lbfgs", random_state=42))
    ])

    logger.info("Fitting TF-IDF + LogisticRegression pipeline...")
    pipeline.fit(X_train, y_train)

    val_preds = pipeline.predict(X_val)
    acc = float(accuracy_score(y_val, val_preds))
    macro_f1 = float(f1_score(y_val, val_preds, average="macro"))

    logger.info(f"Validation Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%")

    # Save artifacts
    model_path = os.path.join(MODEL_DIR, "tfidf_model.pkl")
    label_map_path = os.path.join(MODEL_DIR, "label_map.json")
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    label_map = {label: idx for idx, label in enumerate(RAW_TEXT_LABELS)}
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    metadata = {
        "model_type": "TF-IDF + LogisticRegression",
        "dataset": "dair-ai/emotion + Google GoEmotions (61,627 samples)",
        "num_classes": len(RAW_TEXT_LABELS),
        "raw_classes": RAW_TEXT_LABELS,
        "val_accuracy": acc,
        "val_macro_f1": macro_f1
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved text emotion model artifacts to {MODEL_DIR}")
    return metadata

if __name__ == "__main__":
    train_text_emotion_model()
