import joblib
from typing import Dict
from pathlib import Path


from config import MODEL_BASE_DIR, LABELS


class ToxicityClassifier:
    """A class to handle toxicity classification using a pre-trained model.
    This class loads a model and vectorizer from disk and provides a method to predict toxicity labels for a given text.
    """
    def __init__(self, model_id: str = "latest"):
        """Initialize the ToxicityClassifier with a specific model ID.
        Args:
            model_id: string name of the model (e.g. 'latest', 'model-v1', '2024-07-14-logreg')
        Raises:
            FileNotFoundError: If the model or vectorizer files do not exist.
        """
        self.model_id = model_id
        self._load_model()

    def _load_model(self):
        """
        Load model and vectorizer from disk based on model_id.
        Assumes directory: models/{model_id}/
        The model should be saved as 'toxicity_model.pkl' and the vectorizer as 'vectorizer.pkl'.

        Raises:
            FileNotFoundError: If the model or vectorizer files do not exist.
        """
        model_dir = MODEL_BASE_DIR / self.model_id
        model_path = model_dir / "model.pkl"
        vectorizer_path = model_dir / "vectorizer.pkl"

        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError(f"Model files not found in {model_dir}")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str) -> Dict:
        """
        Predict multilabel toxicity for a given comment.
        Returns a dictionary of labels and boolean predictions.
    
        Args:
            text (str): The comment text to classify.
        Returns:
            Dict: A dictionary with labels as keys and boolean values indicating toxicity.
        """
        X = self.vectorizer.transform([text])
        preds = self.model.predict(X)[0]
        return {label: bool(pred) for label, pred in zip(LABELS, preds)}


class EmbeddingClassifier:
    """A class to handle toxicity classification using a pre-trained embedding model.
    This class loads a model and an embedding encoder from disk and provides a method to predict toxicity
    labels for a given text.
    """
    def __init__(self, model_id: str = "embedding-latest"):
        """
        Loads an embedding-based classifier from disk.
        """
        model_path = Path("models") / model_id
        self.embedder = joblib.load(model_path / "embedder.pkl")
        self.classifier = joblib.load(model_path / "classifier.pkl")
        self.labels = self.classifier.classes_  # o pásalas tú manualmente

    def predict(self, text: str) -> dict:
        """
        Predict toxicity scores from a comment string.
        """
        # Create embedding (batch of size 1)
        embedding = self.embedder.encode([text])
        probs = self.classifier.predict_proba(embedding)[0]  # [0] = single sample
        return dict(zip(self.labels, probs.tolist()))