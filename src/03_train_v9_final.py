import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.lightgbm

def train_final():
    # 1. Load Data
    # Not: Master_credit_final.csv veya v9 hangisini kullanıyorsan yolunu ona göre seç
    data_path = "data/master_credit_v9.csv" 
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    # 2. Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.set_experiment("Credit_Scoring_Final_Robust")
    with mlflow.start_run(run_name="LGBM_Final_V9_Robust"):
        
        # Overfitting-proof parameters
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,    # Lower learning rate for better generalization
            'num_leaves': 31,
            'max_depth': 7,           # Restricted depth
            'lambda_l1': 2.0,         # Strong L1 penalty
            'lambda_l2': 2.0,         # Strong L2 penalty
            'feature_fraction': 0.7,  # Use 70% of features per tree
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'seed': 42,
            'verbose': -1
        }
        
        # Native LightGBM Dataset conversion
        d_train = lgb.Dataset(X_train, label=y_train)
        d_test = lgb.Dataset(X_test, label=y_test, reference=d_train)
        
        # 3. Training with Callbacks
        model = lgb.train(
            params, 
            d_train, 
            valid_sets=[d_train, d_test], 
            valid_names=['train', 'valid'], 
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        # 4. Predictions & Evaluation
        # LightGBM native returns probabilities, we take the max (argmax)
        y_train_pred = model.predict(X_train).argmax(axis=1)
        y_test_pred = model.predict(X_test).argmax(axis=1)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        print("\n" + "="*40)
        print(f"🎯 FINAL ROBUST RESULTS (LGBM)")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Gap (Overfit):  {abs(train_acc - test_acc):.4f}")
        print("="*40)

        # 5. MLflow Logging
        mlflow.log_params(params)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.lightgbm.log_model(model, "robust_lgbm_model")
        
        # 6. Save Model
        joblib.dump(model, "models/credit_model_final_lgbm.joblib")
        print("\n💾 Success! LightGBM model saved.")

if __name__ == "__main__":
    train_final()