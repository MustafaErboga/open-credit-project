import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def build_and_train():
    # 1. Ham Veriyi Yükle
    df = pd.read_csv("data/train.csv", low_memory=False)
    
    # 2. Manuel ve Sabit Temizlik (Hata payı sıfır)
    def force_num(val):
        res = re.findall(r"[-+]?\d*\.?\d+", str(val))
        return float(res[0]) if res else 0.0

    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    
    for col in num_cols:
        df[col] = df[col].apply(force_num)

    # Credit History -> Sabit Formül
    def conv_hist(val):
        p = re.findall(r'\d+', str(val))
        return (int(p[0]) * 12) + int(p[1]) if len(p) >= 2 else 0
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(conv_hist)

    # 3. MANUEL ENCODING (En kritik yer burası, asla değişmez)
    # Poor: 0, Standard: 1, Good: 2
    df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2}).fillna(1)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'No': 0, 'Yes': 1, 'NM': 0.5}).fillna(0.5)
    
    # Rasyolar
    df['DTI'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
    df['EMI_to_Salary'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1)

    # 4. Özellik Seçimi (Sıralama Sabitlendi)
    features = num_cols + ['Credit_History_Months', 'Credit_Mix', 'Payment_of_Min_Amount', 'DTI', 'EMI_to_Salary']
    
    X = df[features].copy()
    y = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})

    # 5. Model Eğitimi (AGRESİF SINIF AĞIRLIĞI)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Poor (0) sınıfına 5 kat, Good (2) sınıfına 3 kat ağırlık veriyoruz!
    weights = {0: 5.0, 1: 1.0, 2: 3.0}

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=10,
        class_weight=weights,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)
    
    print("✅ Model Trained with Atomic Fixes.")
    print(classification_report(y_test, model.predict(X_test)))

    # Modeli ve Sütun Listesini Kaydet (Hata olmasın diye sütunları da gömüyoruz)
    joblib.dump(model, "models/atomic_model.joblib")
    joblib.dump(features, "models/feature_list.joblib")
    print("💾 atomic_model.joblib saved.")

if __name__ == "__main__":
    build_and_train()