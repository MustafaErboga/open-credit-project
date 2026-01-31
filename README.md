---
title: Open Credit Scoring
emoji: 🏦
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# 🏦 OpenCredit: End-to-End Explainable Credit Scoring System

![Python](https://img.shields.io/badge/python-3.11-blue.svg) ![FastAPI]
(https://img.shields.io/badge/FastAPI-v0.128.0-05998b.svg) ![Docker]
(https://img.shields.io/badge/Docker-Enabled-blue.svg) ![MLOps]
(https://img.shields.io/badge/MLOps-MLflow-orange.svg) ![XAI]
(https://img.shields.io/badge/Explainable_AI-SHAP-red.svg)

OpenCredit is a professional-grade, end-to-end Machine Learning project that simulates a real-world financial risk assessment environment. It covers the entire ML lifecycle, from dirty data engineering to cloud-based microservice deployment.

🚀 Live API Documentation
You can interact with the live model via Swagger UI: 👉 Explore API Docs (/docs)

🛠️ Key Features & Engineering Highlights
Leakage-Free Modeling: Unlike common high-scoring Kaggle notebooks, this project is built on a strictly honest pipeline, removing all temporal and ID-based leakage.
Explainable AI (XAI): Integrated with SHAP to ensure model transparency—a legal requirement in the banking sector.
Robust Feature Engineering: Derived high-impact financial ratios such as DTI (Debt-to-Income), EMI-Salary Ratio, and Credit History Months.
Production-Ready Backend: High-performance asynchronous API built with FastAPI and Pydantic for data validation.
Containerization: Fully Dockerized environment to ensure "run-anywhere" portability.
🏗️ Technical Architecture (Lifecycle)
Data Engineering: Cleaned dirty financial records using Regex and handled outliers via Winsorization.
MLOps & Tracking: Experimented with XGBoost, CatBoost, and LightGBM. Tracked all metrics via MLflow.
Model Optimization: Optimized the Champion Model (LightGBM) to achieve a robust 76.40% Accuracy with a narrow 5% Train-Test gap (Generalization).
Deployment: Automated the cloud deployment using Docker on Hugging Face Spaces.
💻 How to Use the API
You can send a POST request to /predict with the following JSON structure:

{
  "Age": 30.0,
  "Annual_Income": 50000.0,
  "Monthly_Inhand_Salary": 4000.0,
  "Num_Bank_Accounts": 2,
  "Num_Credit_Card": 3,
  "Interest_Rate": 10.0,
  "Num_of_Loan": 1,
  "Delay_from_due_date": 2,
  "Num_of_Delayed_Payment": 1,
  "Changed_Credit_Limit": 10.0,
  "Num_Credit_Inquiries": 2,
  "Outstanding_Debt": 1500.0,
  "Credit_History_Months": 120.0,
  "Total_EMI_per_month": 500.0,
  "Amount_invested_monthly": 200.0,
  "Monthly_Balance": 1800.0,
  "DTI": 0.3,
  "EMI_to_Salary": 0.12,
  "Occupation": 1,
  "Payment_Behaviour": 1
}