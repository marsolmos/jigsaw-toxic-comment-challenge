import argparse
import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

import mlflow

from config import RAW_DATA_DIR, MODEL_BASE_DIR, LABELS


def train_embedding_model(model_id: str = "embed-latest"):
    """
    Train an embedding-based model using sentence-transformers and track with MLflow.

    Args:
        model_id (str): Name of the model version to save (e.g., "embed-latest", "embed-v1").
    """
    # 1. Load data
    print("[INFO] Loading training data...")
    df = pd.read_csv(RAW_DATA_DIR / "train.csv")
    df["comment_text"] = df["comment_text"].fillna("").str.strip()

    X = df["comment_text"]
    y = df[LABELS]

    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Create embeddings
    print("[INFO] Generating embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X_train_emb = embedder.encode(X_train.tolist(), show_progress_bar=True)
    X_val_emb = embedder.encode(X_val.tolist(), show_progress_bar=True)

    # 4. Train classifier
    print("[INFO] Training classifier on embeddings...")
    clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000))
    clf.fit(X_train_emb, y_train)

    # 5. Evaluate
    print("[INFO] Evaluating model...")
    y_val_proba = clf.predict_proba(X_val_emb)
    metrics = {}
    for i, label in enumerate(LABELS):
        roc_auc = roc_auc_score(y_val[label], y_val_proba[:, i])
        metrics[label + "_roc_auc"] = roc_auc
        print(f"{label} ROC AUC: {roc_auc:.4f}")

    # Optional: print classification report
    threshold = 0.5
    y_val_pred = (y_val_proba >= threshold).astype(int)
    print("\n[INFO] Classification Report (Validation):")
    print(classification_report(y_val, y_val_pred, target_names=LABELS))

    # 6. Save
    print("[INFO] Saving model...")
    output_dir = MODEL_BASE_DIR / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, output_dir / "classifier.pkl")
    joblib.dump(embedder, output_dir / "embedder.pkl")

    # 7. Log with MLflow
    print("[INFO] Logging with MLflow...")
    with mlflow.start_run(run_name=f"embed_model_{model_id}"):
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("classifier", "LogisticRegression")
        mlflow.log_param("model_id", model_id)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_artifact(output_dir / "classifier.pkl")
        mlflow.log_artifact(output_dir / "embedder.pkl")

    print("[INFO] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train embedding-based classifier.")
    parser.add_argument("--model-id", type=str, default="embed-latest", help="Model version name")
    args = parser.parse_args()

    train_embedding_model(model_id=args.model_id)
