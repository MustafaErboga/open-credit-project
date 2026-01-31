import pandas as pd
import numpy as np
import re

def build_master_v7():
    df = pd.read_csv("data/train.csv", low_memory=False)
    print("🚀 Building V7 Master Data: Incorporating Temporal Behavior...")

    # 1. Temel Temizlik
    def force_numeric(col):
        return pd.to_numeric(df[col].astype(str).str.extract(r'([-+]?\d*\.?\d*)')[0], errors='coerce')

    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Outstanding_Debt', 
                'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

    for col in num_cols:
        df[col] = force_numeric(col)

    # 2. TEMPORAL FEATURES (Zaman İçindeki Değişim)
    # Müşteri bazlı gruplayıp bir önceki aya göre farkları alıyoruz
    df = df.sort_values(by=['Customer_ID', 'Month'])
    
    # Borç değişimi (Geçen aya göre borç arttı mı?)
    df['Debt_Change'] = df.groupby('Customer_ID')['Outstanding_Debt'].diff().fillna(0)
    
    # Kredi limiti değişimi
    df['Limit_Change'] = df.groupby('Customer_ID')['Changed_Credit_Limit'].diff().fillna(0)
    
    # Yatırım alışkanlığı değişimi
    df['Invest_Change'] = df.groupby('Customer_ID')['Amount_invested_monthly'].diff().fillna(0)

    # 3. Kredi Geçmişi Ayı (Regex Fix)
    def conv_hist(val):
        p = re.findall(r'\d+', str(val))
        return (int(p[0]) * 12) + int(p[1]) if len(p) >= 2 else np.nan
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(conv_hist)

    # 4. Rasyolar ve Encoding
    df['DTI'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
    df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2}).fillna(1)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'No': 0, 'Yes': 1, 'NM': 0.5}).fillna(0)

    # 5. Feature Selection
    features = num_cols + ['Credit_History_Months', 'Credit_Mix', 'Payment_of_Min_Amount', 
                           'DTI', 'Debt_change', 'Limit_Change', 'Invest_Change']
    
    # Sütun isimlerini küçük-büyük harf hatasına karşı kontrol et
    df.columns = [c.strip() for c in df.columns]
    
    final_features = [c for c in df.columns if c in features or c == 'Debt_Change']
    
    X = df[final_features].copy()
    X = X.fillna(X.median())
    X['Credit_Score'] = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})
    
    X.to_csv("data/master_credit_v7.csv", index=False)
    print(f"✅ V7 Master Data built: {X.shape}")

if __name__ == "__main__":
    build_master_v7()