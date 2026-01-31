import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

def build_master_v9():
    df = pd.read_csv("data/train.csv", low_memory=False)
    print("🔥 Building THE FINAL MASTER DATA (V9)...")

    # 1. Cleaner Function
    def clean_val(val):
        if pd.isna(val): return np.nan
        res = re.findall(r"[-+]?\d*\.?\d+", str(val))
        return float(res[0]) if res else np.nan

    # 2. Critical Numeric Columns
    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Outstanding_Debt', 
                'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

    for col in num_cols:
        df[col] = df[col].apply(clean_val)

    # 3. Handle 'Credit_History_Age' (The most powerful feature)
    def conv_history(val):
        p = re.findall(r'\d+', str(val))
        return (int(p[0]) * 12) + int(p[1]) if len(p) >= 2 else np.nan
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(conv_history)

    # 4. Handle Categorical Columns (Don't drop them, Label Encode!)
    cat_cols = ['Occupation', 'Payment_Behaviour', 'Credit_Mix', 'Payment_of_Min_Amount']
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 5. Advanced Ratios (Kaggle Winners' Choice)
    df['DTI'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
    df['EMI_to_Salary'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1)
    
    # 6. Target Encoding
    df['Credit_Score'] = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})

    # 7. Final Selection
    features = num_cols + ['Credit_History_Months', 'Occupation', 'Payment_Behaviour', 
                           'Credit_Mix', 'Payment_of_Min_Amount', 'DTI', 'EMI_to_Salary']
    
    X = df[features].copy()
    X = X.fillna(X.median())
    X['Credit_Score'] = df['Credit_Score']
    
    X.to_csv("data/master_credit_v9.csv", index=False)
    print(f"✅ V9 Final Data Created: {X.shape}")

if __name__ == "__main__":
    build_master_v9()