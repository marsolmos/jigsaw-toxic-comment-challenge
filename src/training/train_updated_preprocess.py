import argparse
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

import mlflow

from config import RAW_DATA_DIR, MODEL_BASE_DIR, LABELS
from utils import clean_text


def train_model(model_id: str = "latest", max_features: int = 10000, C: float = 4.0):
    """
    Train a logistic regression model with TF-IDF and track with MLflow.

    Args:
        model_id (str): Name of the model version to save (e.g., "latest", "v1").
        max_features (int): Number of TF-IDF features.
        C (float): Inverse regularization strength for logistic regression.
    """
    # 1. Load training data
    print("[INFO] Loading training data...")
    df = pd.read_csv(RAW_DATA_DIR / "train.csv")

    # 2. Preprocess text
    df["comment_text"] = df["comment_text"].fillna("").apply(clean_text)
    X = df["comment_text"]
    y = df[LABELS]

    # 3. Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Vectorization
    print("[INFO] Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # 5. Train model
    print("[INFO] Training model...")
    clf = OneVsRestClassifier(LogisticRegression(C=C, solver="liblinear"))
    clf.fit(X_train_vec, y_train)

    # 6. Evaluate on validation set
    print("[INFO] Evaluating on validation set...")
    metrics = {}
    y_val_pred_proba = clf.predict_proba(X_val_vec)
    for i, label in enumerate(LABELS):
        y_true = y_val[label]
        y_pred_proba = y_val_pred_proba[:, i]
        score = roc_auc_score(y_true, y_pred_proba)
        metrics[label + "_roc_auc"] = score
        print(f"{label} ROC AUC (val): {score:.4f}")

    # 7. MLflow logging
    print("[INFO] Logging to MLflow...")
    with mlflow.start_run(run_name=f"tfidf_logreg_{model_id}"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("vectorizer", f"TFIDF_{max_features}")
        mlflow.log_param("model_id", model_id)

        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Save artifacts
        output_dir = MODEL_BASE_DIR / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pkl"
        vectorizer_path = output_dir / "vectorizer.pkl"
        joblib.dump(clf, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(vectorizer_path)

    print(f"Model and vectorizer saved under {output_dir}")
    print("MLflow run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train toxicity classifier and log with MLflow.")
    parser.add_argument("--model-id", type=str, default="latest", help="Model version name")
    parser.add_argument("--max-features", type=int, default=10000, help="TF-IDF max features")
    parser.add_argument("--C", type=float, default=4.0, help="Inverse regularization for LogisticRegression")

    args = parser.parse_args()

    train_model(
        model_id=args.model_id,
        max_features=args.max_features,
        C=args.C
    )
