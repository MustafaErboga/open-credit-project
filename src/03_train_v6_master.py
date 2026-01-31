import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import joblib

def train_v6():
    # 1. Veriyi yükle
    df = pd.read_csv("data/master_credit.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    # 2. Stratified Split (Sınıf dengesini korur)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. SMOTE Uygula (Azınlık sınıfları dengeler - Diagnostic Tavsiyesi)
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # 4. MLflow Deneyi
    mlflow.set_experiment("Credit_Scoring_Master")
    with mlflow.start_run(run_name="V6_Master_XGB_SMOTE"):
        
        # Gelişmiş Parametreler
        model = xgb.XGBClassifier(
            n_estimators=700,
            max_depth=9,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist'
        )
        
        model.fit(X_res, y_res)

        # 5. Skorlar
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        print(f"\n🚀 RESULTS")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, model.predict(X_test)))
        
        mlflow.log_metric("test_accuracy", test_acc)
        joblib.dump(model, "models/credit_model_v6.joblib")
        print("💾 Model V6 Saved!")

if __name__ == "__main__":
    train_v6()