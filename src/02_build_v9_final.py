import pandas as pd
import numpy as np
import re

def build_final_master():
    # 1. Ham veriyi yükle
    df = pd.read_csv("data/train.csv", low_memory=False)
    print("🛡️ Building FINAL Safe Master Data (No Leakage Strategy)...")

    # 2. Sayı Temizleme (Regex)
    def clean_num(col):
        return pd.to_numeric(df[col].astype(str).str.extract(r'([-+]?\d*\.?\d*)')[0], errors='coerce')

    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Outstanding_Debt', 
                'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

    for col in num_cols:
        df[col] = clean_num(col)

    # 3. Kredi Geçmişi Yaşı (En dürüst ve güçlü sinyal)
    def conv_hist(val):
        p = re.findall(r'\d+', str(val))
        return (int(p[0]) * 12) + int(p[1]) if len(p) >= 2 else np.nan
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(conv_hist)

    # 4. Rasyolar (Finansal Mantık)
    df['DTI'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
    df['EMI_to_Salary'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1)

    # 5. Kategorik Veriler (Ordinal & Simple Encoding)
    # Credit_Mix en önemli sütunlardan biri
    df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2}).fillna(1)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'No': 0, 'Yes': 1, 'NM': 0.5}).fillna(0)
    
    # Occupation ve Payment_Behaviour (Gürültüyü azaltmak için basit kategori kodlaması)
    df['Occupation'] = df['Occupation'].astype('category').cat.codes
    df['Payment_Behaviour'] = df['Payment_Behaviour'].astype('category').cat.codes

    # 6. Sızıntı Yapabilecek Sütunları ASLA Dahil Etmiyoruz (ID, SSN, Month, Name)
    final_features = num_cols + ['Credit_History_Months', 'DTI', 'EMI_to_Salary', 
                                'Credit_Mix', 'Payment_of_Min_Amount', 'Occupation', 'Payment_Behaviour']
    
    X = df[final_features].copy()
    X = X.fillna(X.median())
    X['Credit_Score'] = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})
    
    # 7. Kaydet
    X.to_csv("data/master_credit_final.csv", index=False)
    print(f"✅ Safe Master Data Created. Shape: {X.shape}")

if __name__ == "__main__":
    build_final_master()