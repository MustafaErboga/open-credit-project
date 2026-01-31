import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

def train_balanced_model():
    df = pd.read_csv("data/train.csv", low_memory=False)
    
    def clean(val):
        res = re.findall(r"[-+]?\d*\.?\d+", str(val))
        return float(res[0]) if res else 0.0

    features = [
        'Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Credit_Mix', 'Annual_Income', 
        'Monthly_Balance', 'Num_Credit_Inquiries', 'Age'
    ]

    for col in ['Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date', 
                'Num_of_Delayed_Payment', 'Annual_Income', 'Monthly_Balance', 'Num_Credit_Inquiries', 'Age']:
        df[col] = df[col].apply(clean)

    # Credit_Mix: Bad=0, Standard=1, Good=2
    df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2}).fillna(1)
    X = df[features].copy()
    y = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- HASSAS AYAR (FINE TUNING) ---
    # Ağırlıkları biraz daha artırıyoruz: 
    # Poor'u koruyoruz (2.5), Good'u iyice belirginleştiriyoruz (4.0)
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,        # Daha da yavaşlattık (daha iyi öğrensin)
        class_weight={0: 2.5, 1: 1.0, 2: 4.0}, 
        max_depth=10,              # Derinliği biraz artırdık
        num_leaves=64,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # --- KOD İÇİ DOĞRULAMA ---
    scenarios = {
        "Zengin (Good)": [50.0, 3.0, 0, 0, 2, 200000.0, 8000.0, 0, 45],
        "Ortalama (Standard)": [1500.0, 15.0, 7, 4, 1, 55000.0, 1200.0, 5, 33],
        "Batık (Poor)": [10000.0, 35.0, 45, 20, 0, 12000.0, 10.0, 15, 20]
    }

    print("\n" + "="*45)
    print("SİSTEM KALİBRASYON TESTİ")
    print("="*45)

    for name, vals in scenarios.items():
        test_df = pd.DataFrame([vals], columns=features)
        pred = model.predict(test_df)[0]
        label = {0: "Poor", 1: "Standard", 2: "Good"}[pred]
        print(f"Senaryo: {name.ljust(20)} -> TAHMİN: {label}")
    
    joblib.dump(model, "models/final_safe_model.joblib")
    joblib.dump(features, "models/final_features.joblib")
    print("="*45)
    print("✅ Model kaydedildi. LÜTFEN UVICORN'U RESTART ET!")

if __name__ == "__main__":
    train_balanced_model()