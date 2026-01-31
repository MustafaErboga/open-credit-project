"""
ADIM 1: Veri Tanılama - Neden 0.80'i Geçemiyoruz?
Bu script veriyi derinlemesine analiz eder ve sorunları tespit eder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("VERİ TANILAMASI - SORUN TESPİTİ")
print("="*60)

# Veriyi yükle
try:
    df = pd.read_csv("data/cleaned_credit.csv")
    print(f"\n✅ Veri yüklendi: {df.shape}")
except FileNotFoundError:
    print("\n❌ cleaned_credit.csv bulunamadı!")
    print("Lütfen veri dosyanızı data/ klasörüne koyun.")
    exit(1)

# 1. Temel İstatistikler
print("\n" + "="*60)
print("1. TEMEL İSTATİSTİKLER")
print("="*60)
print(f"Toplam Satır: {len(df):,}")
print(f"Toplam Sütun: {len(df.columns)}")
print(f"Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. Eksik Veri Kontrolü
print("\n" + "="*60)
print("2. EKSİK VERİ ANALİZİ")
print("="*60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Eksik Sayı': missing,
    'Yüzde (%)': missing_pct
})
missing_df = missing_df[missing_df['Eksik Sayı'] > 0].sort_values('Eksik Sayı', ascending=False)

if len(missing_df) > 0:
    print("⚠️  SORUN: Eksik veriler var!")
    print(missing_df)
else:
    print("✅ Eksik veri yok")

# 3. Hedef Değişken Dağılımı
print("\n" + "="*60)
print("3. HEDEF DEĞİŞKEN DENGESİ (Credit_Score)")
print("="*60)
if 'Credit_Score' in df.columns:
    target_dist = df['Credit_Score'].value_counts()
    target_pct = (target_dist / len(df)) * 100
    
    print("\nDağılım:")
    for class_name, count in target_dist.items():
        pct = target_pct[class_name]
        print(f"  {class_name}: {count:,} ({pct:.1f}%)")
    
    # Class imbalance kontrolü
    max_pct = target_pct.max()
    min_pct = target_pct.min()
    imbalance_ratio = max_pct / min_pct
    
    if imbalance_ratio > 2:
        print(f"\n⚠️  SORUN: Class Imbalance var! Oran: {imbalance_ratio:.2f}x")
        print("   → Çözüm: SMOTE veya class_weight kullanın")
    else:
        print(f"\n✅ Class dengesi iyi (Oran: {imbalance_ratio:.2f}x)")
else:
    print("❌ Credit_Score sütunu bulunamadı!")

# 4. Veri Tipleri
print("\n" + "="*60)
print("4. VERİ TİPİ ANALİZİ")
print("="*60)
dtype_counts = df.dtypes.value_counts()
print("\nVeri Tipleri:")
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} sütun")

# Numeric olmayan ama numeric olması gereken sütunları bul
print("\nProblematik Sütunlar (object tipi ama sayısal olabilir):")
problematic = []
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Credit_Score':  # Hedef değişkeni pas geç
        # İlk birkaç değere bak
        sample = df[col].dropna().head(100)
        # Sayıya çevrilebilir mi?
        try:
            pd.to_numeric(sample, errors='coerce')
            numeric_ratio = sample.apply(lambda x: str(x).replace('.','').replace('-','').isdigit()).sum() / len(sample)
            if numeric_ratio > 0.5:
                problematic.append(col)
                print(f"  ⚠️  {col}: %{numeric_ratio*100:.0f} sayısal görünüyor ama object!")
        except:
            pass

if not problematic:
    print("  ✅ Tüm veri tipleri uygun")

# 5. Outlier Analizi (sadece numeric sütunlar)
print("\n" + "="*60)
print("5. OUTLIER (AYKIRI DEĞER) ANALİZİ")
print("="*60)
numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_summary = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_pct = (outliers / len(df)) * 100
    
    if outlier_pct > 5:  # %5'ten fazla outlier varsa
        outlier_summary.append({
            'Sütun': col,
            'Outlier Sayı': outliers,
            'Yüzde (%)': outlier_pct
        })

if outlier_summary:
    print("⚠️  SORUN: Yüksek oranda outlier var!")
    outlier_df = pd.DataFrame(outlier_summary).sort_values('Yüzde (%)', ascending=False)
    print(outlier_df.to_string(index=False))
    print("\n   → Çözüm: Winsorization veya capping uygulayın")
else:
    print("✅ Outlier oranı kabul edilebilir seviyede")

# 6. Korelasyon Analizi
print("\n" + "="*60)
print("6. KORELASYON ANALİZİ")
print("="*60)

# Target'ı encode et
if 'Credit_Score' in df.columns:
    df_corr = df.copy()
    le = LabelEncoder()
    df_corr['Credit_Score_Encoded'] = le.fit_transform(df_corr['Credit_Score'])
    
    # Sadece numeric sütunlarla korelasyon
    numeric_df = df_corr.select_dtypes(include=[np.number])
    
    if 'Credit_Score_Encoded' in numeric_df.columns:
        target_corr = numeric_df.corr()['Credit_Score_Encoded'].drop('Credit_Score_Encoded')
        target_corr = target_corr.abs().sort_values(ascending=False)
        
        print("\nEn Yüksek Korelasyonlu Özellikler (Top 10):")
        print(target_corr.head(10))
        
        # Düşük korelasyonlu özellikler
        low_corr = target_corr[target_corr < 0.05]
        if len(low_corr) > 0:
            print(f"\n⚠️  SORUN: {len(low_corr)} sütun çok düşük korelasyona sahip (<0.05)")
            print("   Düşük korelasyonlu sütunlar:", low_corr.index.tolist()[:5])
            print("   → Çözüm: Bu sütunları kaldırın veya feature engineering yapın")

# 7. Kardinalite Kontrolü (Kategorik değişkenler)
print("\n" + "="*60)
print("7. KATEGORİK DEĞİŞKEN ANALİZİ")
print("="*60)
categorical_cols = df.select_dtypes(include=['object']).columns
high_cardinality = []

for col in categorical_cols:
    if col != 'Credit_Score':
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df)
        
        if unique_ratio > 0.5:  # %50'den fazla unique değer
            high_cardinality.append({
                'Sütun': col,
                'Unique Sayı': unique_count,
                'Unique Oran': f"{unique_ratio*100:.1f}%"
            })

if high_cardinality:
    print("⚠️  SORUN: Yüksek kardinaliteli kategorik sütunlar var!")
    hc_df = pd.DataFrame(high_cardinality)
    print(hc_df.to_string(index=False))
    print("\n   → Çözüm: Target encoding veya frequency encoding kullanın")
else:
    print("✅ Kategorik değişkenler uygun kardinaliteye sahip")

# 8. Özellik Sayısı
print("\n" + "="*60)
print("8. ÖZELLİK SAYISI ANALİZİ")
print("="*60)
feature_count = len(df.columns) - 1  # Credit_Score hariç
print(f"Toplam Özellik Sayısı: {feature_count}")

if feature_count < 10:
    print("⚠️  SORUN: Çok az özellik var!")
    print("   → Çözüm: Feature engineering ile yeni özellikler türetin")
elif feature_count > 50:
    print("⚠️  SORUN: Çok fazla özellik var!")
    print("   → Çözüm: Feature selection uygulayın")
else:
    print("✅ Özellik sayısı uygun")

# 9. ÖZET ve ÖNERİLER
print("\n" + "="*60)
print("9. ÖZET ve ÖNERİLER")
print("="*60)

issues_found = []
if len(missing_df) > 0:
    issues_found.append("Eksik veriler var")
if 'Credit_Score' in df.columns and (target_pct.max() / target_pct.min()) > 2:
    issues_found.append("Class imbalance var")
if problematic:
    issues_found.append("Yanlış veri tipleri var")
if outlier_summary:
    issues_found.append("Yüksek oranda outlier var")

if issues_found:
    print("\n🔴 Tespit Edilen Sorunlar:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 ÖNERİLER:")
    print("  1. Önce veri temizliği yapın (02_data_cleaning.py)")
    print("  2. Feature engineering uygulayın (03_feature_engineering.py)")
    print("  3. Hyperparameter tuning yapın (04_hyperparameter_tuning.py)")
else:
    print("\n✅ Veri kalitesi iyi görünüyor!")
    print("   Doğrudan feature engineering ve tuning'e geçebilirsiniz.")

print("\n" + "="*60)
print("TANILAIMA TAMAMLANDI")
print("="*60)