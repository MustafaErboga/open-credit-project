import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.xgboost
import joblib
import os

def train_optimized():
    # 1. Load data
    df = pd.read_csv("data/cleaned_credit.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Setup MLflow
    mlflow.set_experiment("Credit_Scoring_Experiment")

    with mlflow.start_run(run_name="XGBoost_Optimized"):
        # 4. Define Advanced Hyperparameters (Inspired by top Kaggle notebooks)
        params = {
            'n_estimators': 300,
            'max_depth': 12,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softmax',
            'num_class': 3,
            'random_state': 42,
            'tree_method': 'hist' # Faster training
        }

        # 5. Initialize and Train XGBoost
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 6. Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 7. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log the model
        mlflow.xgboost.log_model(model, "xgboost_credit_model")

        # 8. Save locally
        joblib.dump(model, "models/credit_model_v2.joblib")

        print("-" * 30)
        print(f"OPTIMIZED XGBOOST SUCCESSFUL")
        print(f"Previous Accuracy: ~0.71")
        print(f"New Accuracy: {acc:.4f}")
        print(f"New F1-Score: {f1:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    train_optimized()