from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict

from training.model import ToxicityClassifier  # relative import

# Initialize FastAPI app
app = FastAPI(
    title="Toxicity Comment Classifier API",
    description="A FastAPI app to classify Wikipedia comments as toxic or non-toxic.",
    version="0.1.0"
)

# Input schema
class Comment(BaseModel):
    text: str


#################################
########## ROOT ENDPOINT ########
#################################
@app.get("/", tags=["Default"])
def root():
    """
    Root endpoint to check if the API is running.
    Returns:
        Dict: A simple message indicating the API is running.
    """
    return {"message": "Toxicity API is running."}


########################################
########## GET AVAILABLE MODELS ########
########################################
@app.get("/models/", tags=["Model Info"])
def list_available_models():
    """
    List available model IDs stored in the model directory.
    """
    from pathlib import Path
    model_dir = Path("models")
    model_ids = [f.name for f in model_dir.iterdir() if f.is_dir()]
    return {"available_models": model_ids}


###################################################
########## PREDICT SINGLE COMMENT ENDPOINT ########
###################################################
@app.post("/predict-single-comment/", tags=["Toxicity Processing"])
def predict_single_comment(comment: Comment, model_id: str = Query("latest")) -> Dict[str, object]:
    """
    Predict toxicity labels for a single comment.

    Args:
        comment (Comment): Comment object with text to classify.
        model_id (str): Identifier of the model version to use.

    Returns:
        Dict: Input, selected model, and prediction results.
    """
    if not comment.text.strip():
        raise HTTPException(status_code=400, detail="Comment text cannot be empty.")

    try:
        clf = ToxicityClassifier(model_id=model_id)
        predictions = clf.predict(comment.text)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    return {
        "model": model_id,
        "input": comment.text,
        "predictions": predictions
    }
