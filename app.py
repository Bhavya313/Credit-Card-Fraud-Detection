from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("rf_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

class FraudRequest(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(request: FraudRequest):
    data = np.array(request.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return {"fraud_prediction": int(prediction[0])}


