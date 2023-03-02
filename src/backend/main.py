import pandas as pd
import numpy as np
import cv2
import io
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from fastapi import FastAPI, File, UploadFile 
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Spin up Mlflow client
client = MlflowClient()

# Get the best model in the experiment used for training by first
# getting all experiments and searching for run with highest accuracy metric
experiments = mlflow.search_experiments(view_type=ViewType.ACTIVE_ONLY)
experiment_ids = [exp.experiment_id for exp in experiments]
runs_df = mlflow.search_runs(
    experiment_ids=experiment_ids,  # List of experiment IDs to search
    run_view_type=ViewType.ACTIVE_ONLY, # View all runs
    order_by=["metrics.val_accuracy DESC"],  # Metrics to sort by and sort order
    max_results=1  # Maximum number of runs to return
)

# Extract the run_id and experiment_id of the top run
run_id = runs_df.iloc[0]["run_id"]
experiment_id = runs_df.iloc[0]["experiment_id"]

# Load best model based on experiment and run ids
model = mlflow.keras.load_model(f"mlruns/{experiment_id}/{run_id}/artifacts/model/")
model_uri = mlflow.get_artifact_uri(f"mlruns/{experiment_id}/{run_id}/artifacts/model/")

def predict_image_class(model, img, threshold=0.6):
    # Load the image
    img = cv2.resize(img, (64, 64))

    # Preprocess the image
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict class probabilities
    preds_proba = model.predict(img)[0]

    # Get class with highest probability
    top_class_idx = np.argmax(preds_proba)

    # Get class with highest probability
    top_class_idx = np.argmax(preds_proba)

    # Check if highest probability is above threshold
    if preds_proba[top_class_idx] > threshold:
        # Return predicted class name
        if top_class_idx == 0:
            return "cat"
        elif top_class_idx == 1:
            return "dog"
        else:
            return "wildlife"
    else:
        return "difficult one to predict, can you provide a better image?"

# Define the predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file and convert it to a numpy array
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert('RGB')
    image = np.asarray(image)
    
    # Make a prediction using the trained model
    prediction = predict_image_class(model, image)

    # Return the prediction result as a JSON response
    return JSONResponse(content=jsonable_encoder({"response": prediction}))