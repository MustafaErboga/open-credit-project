import pandas as pd
import numpy as np
import re

def build_master_pro():
    # 1. Veriyi yükle (Mixed type uyarısını engellemek için low_memory=False)
    df = pd.read_csv("data/train.csv", low_memory=False)
    print("🚀 Master Data Construction Started: Advanced Cleaning Phase...")

    # 2. Sayısal kolonları temizleyen Jilet gibi fonksiyon
    def force_numeric(col_name):
        # Sayı, nokta ve eksi dışındaki her şeyi temizle, sonra sayıya çevir
        # Hatalı olanları (örneğin sadece '_' olanları) NaN yap
        return pd.to_numeric(df[col_name].astype(str).str.extract(r'([-+]?\d*\.?\d*)')[0], errors='coerce')

    # Temizlenmesi gereken tüm sayısal kolonlar
    numeric_features = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance'
    ]

    print("Cleaning numeric columns and handling dirty characters...")
    for col in numeric_features:
        df[col] = force_numeric(col)

    # 3. Mantıksal Sınırlama (Outlier Capping)
    df['Age'] = df['Age'].clip(18, 80)
    df['Num_Bank_Accounts'] = df['Num_Bank_Accounts'].clip(0, 20)
    df['Num_Credit_Card'] = df['Num_Credit_Card'].clip(0, 20)

    # 4. Altın Özellik: Kredi Geçmişi Ayı
    def conv_hist(val):
        if pd.isna(val) or val == 'nan': return np.nan
        try:
            # "22 Years and 5 Months" -> (22*12) + 5
            p = re.findall(r'\d+', str(val))
            if len(p) >= 2:
                return (int(p[0]) * 12) + int(p[1])
            return np.nan
        except: return np.nan
    
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(conv_hist)

    # 5. Kredi Çeşitliliği (Kaggle sırrı)
    df['Num_Loan_Types'] = df['Type_of_Loan'].str.split(',').str.len().fillna(0)

    # 6. Kategorik Kolonları Sayısallaştırma
    df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2}).fillna(1)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'No': 0, 'Yes': 1, 'NM': 0.5}).fillna(0)

    # 7. Finansal Rasyolar
    df['DTI'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
    df['Disposable_Income'] = df['Monthly_Inhand_Salary'] - df['Total_EMI_per_month']

    # 8. Sütun Seçimi ve Eksik Veri Doldurma
    final_features = numeric_features + ['Credit_History_Months', 'Num_Loan_Types', 'Credit_Mix', 'Payment_of_Min_Amount', 'DTI', 'Disposable_Income']
    
    # Sadece seçtiğimiz sütunları al
    X = df[final_features].copy()
    
    # ARTIK BURADA HATA ALMAYACAĞIZ: Çünkü her şey numeric!
    X = X.fillna(X.median())
    
    X['Credit_Score'] = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})
    
    # 9. Kaydet
    X.to_csv("data/master_credit.csv", index=False)
    print(f"✅ Master data successfully built: {X.shape}")

if __name__ == "__main__":
    build_master_pro()