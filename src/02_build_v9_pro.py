import pandas as pd
import numpy as np
import re

def build_pro():
    df = pd.read_csv("data/train.csv", low_memory=False)
    
    def clean(col):
        return pd.to_numeric(df[col].astype(str).str.extract(r'([-+]?\d*\.?\d*)')[0], errors='coerce')

    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                'Outstanding_Debt', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    for col in num_cols: df[col] = clean(col)

    # Credit History
    def conv(val):
        p = re.findall(r'\d+', str(val))
        return (int(p[0]) * 12) + int(p[1]) if len(p) >= 2 else np.nan
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(conv)

    # SABİT ENCODING (API ile tam uyum için)
    df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2}).fillna(1)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'No': 0, 'Yes': 1, 'NM': 0.5}).fillna(0)
    
    # Rasyolar
    df['DTI'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
    df['EMI_to_Salary'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1)

    # Seçim
    X = df[num_cols + ['Credit_History_Months', 'Credit_Mix', 'Payment_of_Min_Amount', 'DTI', 'EMI_to_Salary']].copy()
    X = X.fillna(X.median())
    X['Credit_Score'] = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})
    
    X.to_csv("data/master_credit_pro.csv", index=False)
    print("✅ Pro Data Built (Categories Fixed)")

if __name__ == "__main__":
    build_pro()