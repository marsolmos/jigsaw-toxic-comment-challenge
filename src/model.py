import joblib
from typing import Dict


from config import MODEL_BASE_PATH, LABELS


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
        model_dir = MODEL_BASE_PATH / self.model_id
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
