import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import mlflow
import joblib

def objective(trial):
    # Load data inside objective for optuna parallelization
    df = pd.read_csv("data/cleaned_credit.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameters to tune
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'objective': 'multi:softmax',
        'num_class': 3,
        'tree_method': 'hist',
        'random_state': 42
    }

    model = xgb.XGBClassifier(**param)
    
    # Cross-validation for robust score
    score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3).mean()
    return score

def train_final():
    print("Optimization started with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30) # Set to 20 or 50 for better results

    print("Best Parameters:", study.best_params)

    # Train final model with best params
    df = pd.read_csv("data/cleaned_credit.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Credit_Scoring_Experiment")
    with mlflow.start_run(run_name="XGBoost_Optuna_Final"):
        best_model = xgb.XGBClassifier(**study.best_params)
        best_model.fit(X_train, y_train)

        # Accuracy checks for Overfitting
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        test_acc = accuracy_score(y_test, best_model.predict(X_test))

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        mlflow.log_params(study.best_params)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.xgboost.log_model(best_model, "best_xgb_model")
        
        joblib.dump(best_model, "models/credit_model_final.joblib")
        print("Final Model Saved!")

if __name__ == "__main__":
    train_final()