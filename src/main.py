from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from model import ToxicityClassifier


# Define the API object and load the classifier
app = FastAPI()
classifier = ToxicityClassifier(model_id="latest")

# Define the data model for the comment input
class Comment(BaseModel):
    text: str


#################################
######### ROOT ENDPOINT #########
#################################
@app.get("/", tags=["default"])
def root():
    return {"message": "Toxicity API is running."}


################################################
######### ARTICLE PROCESSING ENDPOINTS #########
################################################
@app.post("/predict-single-comment/", tags=["Toxicity Processing"])
def predict_single_comment(comment: Comment, model_id: str = "latest") -> Dict[str, object]:
    """
    Predict a single comment toxicity labels.
    
    Args:
        comment (Comment): The Wikipedia comment where to predict labels.
        model_id (str): The ID of the model to use for prediction. Defaults to "latest".
        
    Returns:
        Dict: A dictionary containing the classified toxicity labels.
    """
    # Validate the input comment
    if not comment.text:
        raise HTTPException(status_code=400, detail="Comment text cannot be empty.")
    
    # Load the classifier with the specified model ID
    clf = ToxicityClassifier(model_id=model_id)
    predictions = clf.predict(comment.text)

    # Check if predictions are empty
    if not predictions:
        raise HTTPException(status_code=404, detail="Comment could not be classified.")
    return {
        "model": model_id,
        "input": comment.text,
        "predictions": predictions
    }