import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.catboost
import joblib

def train_catboost():
    # 1. Load data
    df = pd.read_csv("data/cleaned_credit.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Setup MLflow
    mlflow.set_experiment("Credit_Scoring_Experiment")

    with mlflow.start_run(run_name="CatBoost_Supreme_V5"):
        
        # 3. Initialize CatBoost (High performance settings)
        model = CatBoostClassifier(
            iterations=1000,
            depth=10,                # Back to deeper trees for intelligence
            learning_rate=0.03,
            l2_leaf_reg=5,           # Strong L2 regularization to prevent overfitting
            random_seed=42,
            verbose=100,             # Shows progress every 100 steps
            eval_metric='Accuracy',
            early_stopping_rounds=50 # Stops if no improvement
        )

        print("Training CatBoost Master Model...")
        model.fit(X_train, y_train, eval_set=(X_test, y_test))

        # 4. Evaluation
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        print("-" * 30)
        print(f"CATBOOST V5 RESULTS")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("-" * 30)

        # 5. Log to MLflow
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.catboost.log_model(model, "catboost_v5_model")

        # 6. Save locally
        joblib.dump(model, "models/credit_model_v5.joblib")
        print("Model v5 saved to models/credit_model_v5.joblib")

if __name__ == "__main__":
    train_catboost()