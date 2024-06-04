# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import logging

# # Initialize FastAPI app
# app = FastAPI()

# # Load the pre-trained model and scaler
# try:
#     model = joblib.load('Models/DBSCAN.joblib')
#     scaler = joblib.load('Models/scaler_updated.joblib')
# except Exception as e:
#     logging.error(f"Error loading model or scaler: {e}")

# # Define input data schema
# class InputData(BaseModel):
#     Number_of_Ratings: float
#     Weighted_Rating: float
#     Rating_category_encoder: int

# # Preprocess input data
# def preprocess_input(input_data: InputData):
#     try:
#         features = [[input_data.Number_of_Ratings, input_data.Weighted_Rating, input_data.Rating_category_encoder]]
#         scaled_features = scaler.transform(features)
#         return scaled_features
#     except Exception as e:
#         logging.error(f"Error during preprocessing: {e}")
#         raise HTTPException(status_code=500, detail="Error during preprocessing")

# # Prediction endpoint
# @app.post("/predict")
# async def predict(input_data: InputData):
#     try:
#         scaled_features = preprocess_input(input_data)
#         cluster_labels = model.predict(scaled_features)
#         return {"cluster_labels": cluster_labels.tolist()}
#     except Exception as e:
#         logging.error(f"Error during prediction: {e}")
#         raise HTTPException(status_code=500, detail="Error during prediction")

# # Root endpoint
# @app.get("/")
# def root():
#     return {"message": "Welcome to my FastAPI application!"}

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# Load your model
model_dir = os.path.dirname(__file__)
kmeans_model = joblib.load(os.path.join(model_dir, 'models/K-Means.joblib'))
dbscan_model = joblib.load(os.path.join(model_dir, 'models/DBSCAN.joblib'))
scaler = joblib.load(os.path.join(model_dir, 'models/scaler.joblib'))

class PredictionRequest(BaseModel):
   Number_of_Ratings: float
   Weighted_Rating: float
   Rating_category_encoder: int

@app.post("/predict/kmeans")
async def predict_kmeans(data: PredictionRequest):
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    scaled_data = scaler.transform(df)
    prediction = kmeans_model.predict(scaled_data)
    return {"cluster": int(prediction[0])}

@app.post("/predict/dbscan")
async def predict_dbscan(data: PredictionRequest):
    df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    scaled_data = scaler.transform(df)
    prediction = dbscan_model.fit_predict(scaled_data)
    cluster_label = int(prediction[0]) if prediction.size > 0 else -1  # Handle case where no cluster is assigned
    return {"cluster": cluster_label}
