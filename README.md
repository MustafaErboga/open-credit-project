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

[![FastAPI](https://img.shields.io/badge/FastAPI-v0.111.0-05998b.svg)](https://fastapi.tiangolo.com/) 
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/) 
[![MLOps](https://img.shields.io/badge/MLOps-MLflow-orange.svg)](https://mlflow.org/) 
[![XAI](https://img.shields.io/badge/Explainable_AI-SHAP-red.svg)](https://shap.readthedocs.io/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/MustafaErboga/open-credit-scoring)

OpenCredit is a professional-grade, end-to-end Machine Learning project that simulates a real-world financial risk assessment environment. It covers the entire ML lifecycle, from advanced data engineering to cloud-based microservice deployment.

## 🔗 Live Demo & API Docs
You can interact with the live model hosted on Hugging Face Spaces:
👉 **[OpenCredit Live API Docs (/docs)](https://huggingface.co/spaces/MustafaErboga/open-credit-scoring/docs)**

---

## 🛠️ Key Features & Engineering Highlights
*   **Leakage-Free Modeling:** Built on a strictly honest pipeline by removing all temporal and ID-based leakage (Customer_ID, SSN, Month) to ensure real-world reliability.
*   **Explainable AI (XAI):** Integrated with **SHAP** to provide transparent credit decisions, meeting the strict legal requirements of the banking sector.
*   **CI/CD Automation:** Fully integrated workflow between **GitHub** and **Hugging Face Spaces**. Any push to the `main` branch triggers an automated build and deployment process.
*   **Git LFS (Large File Storage):** Professional management of large binary model files (`.joblib`) using Git LFS, ensuring version control integrity.
*   **Robust Preprocessing:** Handled extreme outliers via Winsorization and balanced the majority class bias using custom class weighting (Inverse Ratio Scaling).

---

## 🏗️ Technical Architecture (Lifecycle)
1.  **Data Engineering:** Cleaned 100k records using Regex and domain-logic clipping (Winsorization).
2.  **MLOps & Tracking:** Experimented with XGBoost, CatBoost, and LightGBM while tracking all hyperparameters and metrics via **MLflow**.
3.  **Model Calibration:** Tuned the champion **LightGBM** model to achieve a robust **76.40% Accuracy** with a narrow 5% Train-Test gap to ensure high generalization.
4.  **Containerization:** Fully Dockerized using a specialized Linux base (`python:3.10-slim`) with `libgomp1` dependencies for high-performance inference.

---

## 💻 How to Use the API
Send a **POST** request to `/predict` with the following 9 high-impact features:

**Request Body Example:**
```json
{
  "Outstanding_Debt": 1200.0,
  "Interest_Rate": 12.0,
  "Delay_from_due_date": 5,
  "Num_of_Delayed_Payment": 3,
  "Credit_Mix": 1,
  "Annual_Income": 55000.0,
  "Monthly_Balance": 1500.0,
  "Num_Credit_Inquiries": 4,
  "Age": 32.0
}
```

**Response Example:**
```
{
  "prediction": "Standard",
  "confidence_score": 0.5626,
  "probabilities": {
    "Poor": 0.3091,
    "Standard": 0.5626,
    "Good": 0.1283
  }
}
```

---

## 📂 Project Structure
*   `app.py`: FastAPI server with calibrated inference logic and HTML landing page.
*   `src/`: Preprocessing and training scripts (Evolution from V1 to Final Master).
*   `models/`: Serialized model and feature artifacts.
*   `notebooks/`: Exploratory Data Analysis and SHAP visualizations.
*   `Dockerfile`: Container configuration for global deployment.
*   `requirements.txt`: Project dependencies.

---

## 🚀 Local Setup & Installation

### 1. Clone the Repository
```
git clone https://github.com/MustafaErboga/open-credit-project.git
cd open-credit-project
```

### 2. Run with Docker (Recommended)
```
docker build -t open-credit-api .
docker run -p 8000:7860 open-credit-api
```
*Access the API at `http://localhost:8000/docs`*

### 3. Manual Installation
```
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

---

## 📊 Evaluation Scenarios
The model has been strictly validated against three critical financial profiles:
*   **High-Net-Worth:** Low debt, high income, long history → Predicted: **GOOD** (High Confidence)
*   **Risk Profile:** Low income, high debt, multiple delays → Predicted: **POOR** (High Sensitivity)
*   **Standard:** Balanced income/debt ratios → Predicted: **STANDARD** (Stable)
```
