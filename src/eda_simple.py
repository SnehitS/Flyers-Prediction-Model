"""Simple EDA without plotting - fast version."""

import pandas as pd
import numpy as np

csv_path = r'c:\Users\shada\flyers-prediction-model\data\processed\features.csv'

print("Loading data...")
df = pd.read_csv(csv_path)

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print(f"\nDataset Shape: {df.shape}")
print(f"Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")

print(f"\n--- Missing Values (Top 15) ---")
missing = df.isnull().sum().sort_values(ascending=False)
for col, count in missing.head(15).items():
    pct = 100.0 * count / len(df)
    if count > 0:
        print(f"  {col:30s}: {count:4d} ({pct:5.1f}%)")

print(f"\n--- Label Distribution ---")
print(f"  Home wins: {(df['home_win'] == 1).sum()}")
print(f"  Home losses: {(df['home_win'] == 0).sum()}")
print(f"  Home win rate: {df['home_win'].mean():.1%}")

print(f"\n--- Feature Overview ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'home_win' in numeric_cols:
    numeric_cols.remove('home_win')
print(f"  Total numeric features: {len(numeric_cols)}")

print(f"\n  Features with 100% data:")
for col in numeric_cols:
    pct = 100.0 * df[col].notna().sum() / len(df)
    if pct == 100:
        print(f"    - {col}")

print(f"\n  Features with partial data:")
for col in numeric_cols:
    pct = 100.0 * df[col].notna().sum() / len(df)
    if 0 < pct < 100:
        print(f"    - {col}: {pct:.1f}%")

print(f"\n  Features with no data:")
for col in numeric_cols:
    pct = 100.0 * df[col].notna().sum() / len(df)
    if pct == 0:
        print(f"    - {col}")

print(f"\n--- Top Features by Correlation with home_win ---")
correlations = df[numeric_cols + ['home_win']].corr()['home_win'].drop('home_win').abs().sort_values(ascending=False)
for col, corr in correlations.head(15).items():
    print(f"  {col:30s}: {corr:+.4f}")

print(f"\n--- Feature Statistics ---")
stats = df[numeric_cols].describe().T
print(f"Count: {len(numeric_cols)} features")
print(f"\nTop 5 features by variance:")
variance = df[numeric_cols].var().sort_values(ascending=False)
for col, var in variance.head(5).items():
    mean = df[col].mean()
    print(f"  {col:30s}: mean={mean:8.2f}, var={var:10.2f}")

print("\n" + "=" * 70)
print("EDA Complete!")
print("=" * 70)
