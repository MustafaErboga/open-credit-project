import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import joblib

def train_v8():
    df = pd.read_csv("data/master_credit_v8.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.set_experiment("Credit_Scoring_Final")
    with mlflow.start_run(run_name="CatBoost_V8_Pro"):
        
        # High-Performance Settings
        model = CatBoostClassifier(
            iterations=2000,          # More iterations
            learning_rate=0.05,       # Slightly higher learning
            depth=10,                 # Deep enough to learn patterns
            l2_leaf_reg=3,            # Reduced regularization to increase learning
            auto_class_weights='Balanced', # Better than SMOTE for CatBoost
            random_seed=42,
            verbose=100,
            early_stopping_rounds=100
        )
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test))

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        print(f"\n🏆 V8 PRO RESULTS")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\n", classification_report(y_test, model.predict(X_test)))
        
        joblib.dump(model, "models/credit_model_v8.joblib")
        print("💾 Master Model V8 Saved!")

if __name__ == "__main__":
    train_v8()