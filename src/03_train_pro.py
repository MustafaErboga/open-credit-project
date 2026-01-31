import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_pro():
    df = pd.read_csv("data/master_credit_pro.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # DİKKAT: Poor (0) ağırlığını artırdık, Good (2) ağırlığını biraz dengeledik.
    # Bu ayar, modelin "batık" müşteriye "Standard" demesini engelleyecek.
    class_weights = {0: 3.5, 1: 1.0, 2: 2.0} 

    model = lgb.LGBMClassifier(
        n_estimators=1500,         # Daha fazla ağaç ile detayları yakala
        learning_rate=0.02,        # Daha yavaş ve sağlam öğren
        max_depth=12,              # Biraz daha derinleş
        num_leaves=64,             # Karar ağacını genişlet
        class_weight=class_weights,
        reg_alpha=0.5,             # Regularization (Gürültüyü sil)
        reg_lambda=0.5,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
              callbacks=[lgb.early_stopping(stopping_rounds=100)])

    print(f"🎯 NEW TEST ACCURACY: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    joblib.dump(model, "models/credit_model_pro.joblib")
    print("💾 Model calibrated and saved!")

if __name__ == "__main__":
    train_pro()