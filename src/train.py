import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import joblib
import os

def train_model():
    # 1. Load the cleaned dataset
    data_path = "data/cleaned_credit.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run the cleaning notebook first.")
        return

    df = pd.read_csv(data_path)

    # 2. Split features and target
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize MLflow Experiment
    mlflow.set_experiment("Credit_Scoring_Experiment")

    with mlflow.start_run(run_name="Random_Forest_Baseline"):
        # Define model parameters
        n_estimators = 100
        max_depth = 10
        
        # Initialize and train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.fit(X_train, y_train).predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # 5. Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # 6. Log the model artifact
        mlflow.sklearn.log_model(model, "credit_scoring_model")

        # 7. Save the model locally for later use
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, "credit_model.joblib")
        joblib.dump(model, model_path)

        print(f"Successfully trained model with Accuracy: {acc:.4f}")
        print(f"Model saved to {model_path}")
        print("Check MLflow UI (http://127.0.0.1:5000) for detailed logs.")

if __name__ == "__main__":
    train_model()