import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import joblib
import os

def train_ensemble():
    # 1. Load data
    df = pd.read_csv("data/cleaned_credit.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Setup MLflow
    mlflow.set_experiment("Credit_Scoring_Experiment")

    with mlflow.start_run(run_name="Ensemble_Voting_V4"):
        
        # MODEL 1: Optimized XGBoost (Preventing Overfitting)
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,             # Reduced depth to prevent memorization
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,            # L1 Regularization
            reg_lambda=1,           # L2 Regularization
            random_state=42
        )

        # MODEL 2: LightGBM (Fast and robust)
        lgb_model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=-1
        )

        # MODEL 3: Random Forest (Different architecture)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )

        # 3. Create Voting Classifier (The Ensemble)
        ensemble_model = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)],
            voting='soft' # Uses probabilities for better decisions
        )

        print("Training Ensemble Model (XGB + LGBM + RF)...")
        ensemble_model.fit(X_train, y_train)

        # 4. Evaluation (Crucial Check for Overfitting)
        train_preds = ensemble_model.predict(X_train)
        test_preds = ensemble_model.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        print("-" * 30)
        print(f"ENSEMBLE MODEL V4 RESULTS")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("-" * 30)

        # 5. Log to MLflow
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_param("ensemble_types", "XGB_LGBM_RF")
        
        mlflow.sklearn.log_model(ensemble_model, "ensemble_v4_model")

        # 6. Save locally
        joblib.dump(ensemble_model, "models/credit_model_v4.joblib")
        print("Model v4 saved to models/credit_model_v4.joblib")

if __name__ == "__main__":
    train_ensemble()