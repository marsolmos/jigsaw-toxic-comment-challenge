from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict

from src.model import ToxicityClassifier  # relative import
from src.config import LABELS  # optionally used for validation or response schema

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
    return {"message": "Toxicity API is running."}


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
