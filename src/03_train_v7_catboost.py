import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import joblib

def train_v7():
    df = pd.read_csv("data/master_credit_v7.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    mlflow.set_experiment("Credit_Scoring_Final")
    with mlflow.start_run(run_name="CatBoost_V7_Temporal"):
        
        # Ezberlemeyi önleyen (Overfitting-proof) Parametreler
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=10,        # Yüksek ceza puanı (ezberlemeyi önler)
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50
        )
        
        model.fit(X_res, y_res, eval_set=(X_test, y_test))

        test_preds = model.predict(X_test)
        train_preds = model.predict(X_train)

        print(f"\n🎯 FINAL MASTER RESULTS")
        print(f"Train Accuracy: {accuracy_score(y_train, train_preds):.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.4f}")
        print("\n", classification_report(y_test, test_preds))
        
        joblib.dump(model, "models/credit_model_v7.joblib")
        print("💾 Master Model V7 Saved!")

if __name__ == "__main__":
    train_v7()