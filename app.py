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
        <head>
            <title>OpenCredit API</title>
            <style>
                body { 
                    font-family: 'Segoe UI', sans-serif; 
                    background-color: #0d1117; 
                    color: #ffffff; 
                    text-align: center; 
                    padding: 100px 20px;
                    margin: 0;
                }
                .card {
                    background-color: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 12px;
                    padding: 40px;
                    display: inline-block;
                    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
                    max-width: 550px;
                }
                h1 { color: #238636; margin-bottom: 10px; }
                p { color: #8b949e; font-size: 1.1em; line-height: 1.6; }
                .status-badge {
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 20px;
                    background-color: rgba(35, 134, 54, 0.2);
                    color: #3fb950;
                    font-size: 0.85em;
                    font-weight: 600;
                    border: 1px solid rgba(63, 185, 80, 0.3);
                    margin-bottom: 20px;
                }
                .btn {
                    display: inline-block;
                    margin-top: 25px;
                    padding: 12px 30px;
                    background-color: #238636;
                    color: white;
                    text-decoration: none;
                    border-radius: 6px;
                    font-weight: 600;
                    transition: 0.2s;
                }
                .btn:hover { background-color: #2ea043; }
                .footer { margin-top: 60px; font-size: 0.8em; color: #484f58; }
            </style>
        </head>
        <body>
            <div class="card">
                <div class="status-badge">● System Online</div>
                <h1>OpenCredit Scoring System</h1>
                <p>An end-to-end explainable credit risk assessment API powered by LightGBM and SHAP.</p>
                <p>This service provides real-time credit scoring and probability distributions for financial risk management.</p>
                
                <a href="/docs" class="btn">Explore API Documentation</a>
            </div>
            <div class="footer">
                Developed by Mustafa Erboga | MLOps Engineering Project
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