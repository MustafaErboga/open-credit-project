import pandas as pd
import numpy as np
import re

def build_master_v8():
    df = pd.read_csv("data/train.csv", low_memory=False)
    print("🚀 Building V8 Master Data: The Professional Standard...")

    # 1. Numeric Extraction (Advanced Regex)
    def force_numeric(col):
        return pd.to_numeric(df[col].astype(str).str.extract(r'([-+]?\d*\.?\d*)')[0], errors='coerce')

    num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Outstanding_Debt', 
                'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

    for col in num_cols:
        df[col] = force_numeric(col)

    # 2. Credit History Age (Crucial for 90%+)
    def conv_hist(val):
        p = re.findall(r'\d+', str(val))
        return (int(p[0]) * 12) + int(p[1]) if len(p) >= 2 else np.nan
    df['Credit_History_Months'] = df['Credit_History_Age'].apply(conv_hist)

    # 3. Type of Loan (Counting Specific Importance)
    df['Num_Loan_Types'] = df['Type_of_Loan'].str.split(',').str.len().fillna(0)

    # 4. Behavioral Features (Temporal)
    df = df.sort_values(by=['Customer_ID', 'Month'])
    df['Debt_Change'] = df.groupby('Customer_ID')['Outstanding_Debt'].diff().fillna(0)
    df['Monthly_Balance_Change'] = df.groupby('Customer_ID')['Monthly_Balance'].diff().fillna(0)

    # 5. Financial Ratios (Feature Engineering)
    df['DTI'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
    df['EMI_Salary_Ratio'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1)

    # 6. Encoding
    df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2}).fillna(1)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'No': 0, 'Yes': 1, 'NM': 0.5}).fillna(0)

    # 7. Final Selection
    features = num_cols + ['Credit_History_Months', 'Num_Loan_Types', 'Debt_Change', 
                           'Monthly_Balance_Change', 'DTI', 'EMI_Salary_Ratio', 
                           'Credit_Mix', 'Payment_of_Min_Amount']
    
    X = df[features].copy()
    X = X.fillna(X.median())
    X['Credit_Score'] = df['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})
    
    X.to_csv("data/master_credit_v8.csv", index=False)
    print(f"✅ V8 Master Data Ready. Shape: {X.shape}")

if __name__ == "__main__":
    build_master_v8()