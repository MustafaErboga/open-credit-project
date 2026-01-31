import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

def final_fix():
    # 1. Veriyi Oku
    df = pd.read_csv("data/train.csv", low_memory=False)
    
    # 2. Temizlik (Sadece en kritik 10 sütun - Kafa karışıklığına son)
    def clean(val):
        res = re.findall(r"[-+]?\d*\.?\d+", str(val))
        return float(res[0]) if res else 0.0

    # Modelin bakacağı "Gerçek Risk" sütunları
    features = [
        'Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Credit_Mix', 'Annual_Income', 
        'Monthly_Balance', 'Num_Credit_Inquiries', 'Age'
    ]

    # Sayısallaştırma
    for col in ['Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date', 
                'Num_of_Delayed_Payment', 'Annual_Income', 'Monthly_Balance', 'Num_Credit_Inquiries', 'Age']:
        df[col] = df[col].apply(clean)

    # Credit_Mix: Bad=0, Standard=1, Good=2
    df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2}).fillna(0)

    X = df[features].copy()
    y = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})

    # 3. Model Eğitimi (POOR Sınıfına Dev Ağırlık)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Poor (0) sınıfına 10 kat, Standard'a (1) yarım kat ağırlık!
    # Bu ayar modeli "Poor olanı kaçırırsan biteriz" diye zorlar.
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        class_weight={0: 10.0, 1: 0.5, 2: 5.0}, 
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)

    # 4. KODUN İÇİNDE TEST (Batık Adam Senaryosu)
    batik_adam = pd.DataFrame([{
        'Outstanding_Debt': 15000.0,  # Devasa borç
        'Interest_Rate': 35.0,        # Devasa faiz
        'Delay_from_due_date': 60,    # 2 ay gecikme
        'Num_of_Delayed_Payment': 30, # Sürekli gecikme
        'Credit_Mix': 0,              # Bad mix
        'Annual_Income': 5000.0,      # Çok düşük gelir
        'Monthly_Balance': 10.0,      # Metelik yok
        'Num_Credit_Inquiries': 20,   # Her yerden kredi istemiş
        'Age': 20.0
    }])

    prediction = model.predict(batik_adam)[0]
    label = {0: "Poor", 1: "Standard", 2: "Good"}[prediction]
    
    print("\n" + "="*30)
    print(f"TEST SONUCU (Batik Adam): {label}")
    print("="*30)

    if label == "Poor":
        joblib.dump(model, "models/final_safe_model.joblib")
        joblib.dump(features, "models/final_features.joblib")
        print("✅ Başardık! Model Poor dedi. Kaydedildi.")
    else:
        print("❌ Hala Standard diyor. Parametreleri daha da sertleştirmem lazım.")

if __name__ == "__main__":
    final_fix()