import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow

def train_v9():
    # 1. Load Data
    df = pd.read_csv("data/master_credit_v9.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.set_experiment("Credit_Scoring_Final_V9")
    with mlflow.start_run(run_name="XGBoost_V9_Final_Check"):
        
        # FIX: early_stopping_rounds is now defined here, not in .fit()
        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=2,
            random_state=42,
            tree_method='hist',
            early_stopping_rounds=50  # <--- Moved here
        )
        
        # 2. Training (Removed early_stopping_rounds from here)
        model.fit(
            X_train, y_train, 
            eval_set=[(X_test, y_test)], 
            verbose=100
        )

        # 3. Predictions for Gap Check
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 4. Accuracy Calculations
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        print("\n" + "="*30)
        print(f"🎯 V9 FINAL PERFORMANCE")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Gap (Overfit):  {abs(train_acc - test_acc):.4f}")
        print("="*30)

        # MLflow Logs
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        
        # Save model
        joblib.dump(model, "models/credit_model_v9.joblib")
        print("\n💾 Success! Model saved as models/credit_model_v9.joblib")

if __name__ == "__main__":
    train_v9()