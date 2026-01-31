import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def build_and_train():
    # 1. Veriyi Oku
    df = pd.read_csv("data/train.csv", low_memory=False)
    print("🚀 Ultimate Pipeline Started...")

    # 2. Temizlik Fonksiyonu
    def clean_num(val):
        if pd.isna(val): return 0.0
        res = re.findall(r"[-+]?\d*\.?\d+", str(val))
        return float(res[0]) if res else 0.0

    # 3. Sadece En Güçlü Özellikleri Seçiyoruz (Noise'u engellemek için)
    # Bu özellikler SHAP grafiğinde en tepede çıkanlar
    target_cols = [
        'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 
        'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
        'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Outstanding_Debt', 
        'Monthly_Balance', 'Total_EMI_per_month'
    ]
    
    for col in target_cols:
        df[col] = df[col].apply(clean_num)

    # Credit History -> Months
    def conv_hist(val):
        p = re.findall(r'\d+', str(val))
        return (int(p[0]) * 12) + int(p[1]) if len(p) >= 2 else 0
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(conv_hist)

    # 4. MANUEL ENCODING (API ile %100 uyum garantisi)
    df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2}).fillna(1)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'No': 0, 'Yes': 1, 'NM': 0.5}).fillna(0.5)

    # 5. Final Özellik Listesi
    features = target_cols + ['Credit_History_Months', 'Credit_Mix', 'Payment_of_Min_Amount']
    
    X = df[features].copy()
    y = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})

    # Eksikleri doldur
    X = X.fillna(X.median())

    # 6. Eğitim ve Test Ayırımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 7. Model: Dengeli LightGBM
    # Sınıf ağırlıklarını Standard'ı (1) baskılayacak şekilde ayarlıyoruz
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=31,
        class_weight={0: 3.0, 1: 1.0, 2: 3.0}, # Poor ve Good'u Standard'dan daha önemli yap!
        random_state=42,
        verbose=-1
    )

    print("Training...")
    model.fit(X_train, y_train)

    # 8. Değerlendirme
    y_pred = model.predict(X_test)
    print("\n✅ Final Model Success Rate:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))

    # 9. Kaydet
    joblib.dump(model, "models/ultimate_model.joblib")
    joblib.dump(features, "models/ultimate_features.joblib")
    print("💾 Model and Feature List saved successfully!")

if __name__ == "__main__":
    build_and_train()