from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="OpenCredit: Transparent Scoring API")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
        <head><title>OpenCredit API</title></head>
        <body style="font-family: sans-serif; text-align: center; padding-top: 100px; background-color: #0d1117; color: white;">
            <div style="display: inline-block; padding: 40px; background: #161b22; border-radius: 12px; border: 1px solid #30363d;">
                <h1 style="color: #238636;">OpenCredit Scoring System</h1>
                <p style="color: #8b949e;">System Online - Stable Version</p>
                <a href="/docs" style="display: inline-block; margin-top: 20px; padding: 10px 20px; background: #238636; color: white; text-decoration: none; border-radius: 6px;">Explore API Docs</a>
            </div>
        </body>
    </html>
    """

model = joblib.load("models/final_safe_model.joblib")
features = joblib.load("models/final_features.joblib")

class Customer(BaseModel):
    Outstanding_Debt: float
    Interest_Rate: float
    Delay_from_due_date: int
    Num_of_Delayed_Payment: int
    Credit_Mix: int
    Annual_Income: float
    Monthly_Balance: float
    Num_Credit_Inquiries: int
    Age: float

@app.post("/predict")
def predict(data: Customer):
    input_df = pd.DataFrame([data.dict()])[features]
    probs = model.predict_proba(input_df)[0]
    res_index = int(np.argmax(probs))
    class_map = {0: "Poor", 1: "Standard", 2: "Good"}
    
    return {
        "prediction": class_map[res_index],
        "confidence_score": round(float(np.max(probs)), 4),
        "probabilities": {
            "Poor (Class 0)": round(float(probs[0]), 4),
            "Standard (Class 1)": round(float(probs[1]), 4),
            "Good (Class 2)": round(float(probs[2]), 4)
        }
    }