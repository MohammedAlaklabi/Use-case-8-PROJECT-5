from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model and scaler
try:
    model = joblib.load('Models/K-Means.joblib')
    scaler = joblib.load('Models/scaler.joblib')
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")

# Define input data schema
class InputData(BaseModel):
    Number_of_Ratings: float
    Weighted_Rating: float
    Rating_category_encoder: int

# Preprocess input data
def preprocess_input(input_data: InputData):
    try:
        features = [[input_data.Number_of_Ratings, input_data.Weighted_Rating, input_data.Rating_category_encoder]]
        scaled_features = scaler.transform(features)
        return scaled_features
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise HTTPException(status_code=500, detail="Error during preprocessing")

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    try:
        scaled_features = preprocess_input(input_data)
        cluster_labels = model.predict(scaled_features)
        return {"cluster_labels": cluster_labels.tolist()}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")

# Root endpoint
@app.get("/")
def root():
    return "Welcome to my FastAPI application!"
