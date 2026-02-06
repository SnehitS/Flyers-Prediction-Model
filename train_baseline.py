#!/usr/bin/env python3
"""Minimal baseline model - fast local training."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("BASELINE MODEL TRAINING")
print("=" * 70)

# Load data
print("\nLoading data...")
df = pd.read_csv("data/processed/features.csv")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")

# Select features (exclude NaN columns and target-related columns)
exclude_cols = {'gamePk', 'gameDate', 'home_win', 'away_win', 'result_type', 'home_score', 'away_score',
                'home_wins', 'home_losses', 'away_wins', 'away_losses'}
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
               if col not in exclude_cols]

# Select only features with <50% missing values
valid_cols = [col for col in numeric_cols if df[col].isnull().sum() / len(df) < 0.5]
X = df[valid_cols].copy()
y = df['home_win'].copy()

# Fill remaining NaNs with median
for col in X.columns:
    X[col].fillna(X[col].median(), inplace=True)

# Verify no NaNs remain
if X.isnull().any().any():
    print(f"WARNING: NaN values still present in features: {X.columns[X.isnull().any()].tolist()}")
    valid_rows = ~X.isnull().any(axis=1)
    X = X[valid_rows]
    y = y[valid_rows]

print(f"  Features selected: {len(X.columns)}")
print(f"  Samples: {len(X)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Test:  {X_test.shape[0]} samples")
print(f"  Home win rate: {y.mean():.1%}")

# Model 1: Logistic Regression
print("\n" + "-" * 70)
print("Model 1: Logistic Regression")
print("-" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
model_lr.fit(X_train_scaled, y_train)

y_pred_lr = model_lr.predict(X_test_scaled)
y_proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_proba_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"  Accuracy:  {acc_lr:.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_lr):.4f}")
print(f"  F1 Score:  {f1_lr:.4f}")
print(f"  AUC:       {auc_lr:.4f}")

# Model 2: Random Forest
print("\n" + "-" * 70)
print("Model 2: Random Forest")
print("-" * 70)

model_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
y_proba_rf = model_rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"  Accuracy:  {acc_rf:.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"  F1 Score:  {f1_rf:.4f}")
print(f"  AUC:       {auc_rf:.4f}")

# Comparison
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(f"{'Model':<20} {'Accuracy':>12} {'AUC':>12} {'F1':>12}")
print("-" * 70)
print(f"{'Logistic Regression':<20} {acc_lr:>12.4f} {auc_lr:>12.4f} {f1_lr:>12.4f}")
print(f"{'Random Forest':<20} {acc_rf:>12.4f} {auc_rf:>12.4f} {f1_rf:>12.4f}")
print("=" * 70)

# Feature importance (Random Forest)
print("\nTop 10 Most Important Features (Random Forest):")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

print("\n" + "=" * 70)
print("Complete! Models trained and compared.")
print("=" * 70)
