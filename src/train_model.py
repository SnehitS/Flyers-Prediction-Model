"""Train baseline models for NHL home_win prediction."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
import json


def prepare_data(csv_path: str = "data/processed/features.csv", test_size: float = 0.2, random_state: int = 42):
    """Load and prepare data for modeling."""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Select numeric features, excluding ID/date and target
    exclude_cols = {'gamePk', 'gameDate', 'home_win', 'away_win', 'result_type', 'home_score', 'away_score'}
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    print(f"Features selected: {len(numeric_cols)}")
    
    # Create feature matrix and target
    X = df[numeric_cols].copy()
    y = df['home_win'].copy()
    
    # Drop rows with missing targets
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"After removing missing targets: {X.shape[0]} samples")
    
    # Fill missing feature values with median
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    print(f"Missing values after filling: {X.isnull().sum().sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Home win rate (train): {y_train.mean():.1%}")
    print(f"Home win rate (test): {y_test.mean():.1%}")
    
    return X_train, X_test, y_train, y_test, numeric_cols


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train logistic regression baseline."""
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 70)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training logistic regression...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    results = {
        'model': 'Logistic Regression',
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test),
        'test_recall': recall_score(y_test, y_pred_test),
        'test_f1': f1_score(y_test, y_pred_test),
        'test_auc': roc_auc_score(y_test, y_proba_test),
    }
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
    print(f"  Test Precision: {results['test_precision']:.4f}")
    print(f"  Test Recall:    {results['test_recall']:.4f}")
    print(f"  Test F1 Score:  {results['test_f1']:.4f}")
    print(f"  Test AUC:       {results['test_auc']:.4f}")
    
    return model, scaler, results, y_pred_test, y_proba_test


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost baseline."""
    print("\n" + "=" * 70)
    print("XGBOOST BASELINE")
    print("=" * 70)
    
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    results = {
        'model': 'XGBoost',
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test),
        'test_recall': recall_score(y_test, y_pred_test),
        'test_f1': f1_score(y_test, y_pred_test),
        'test_auc': roc_auc_score(y_test, y_proba_test),
    }
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
    print(f"  Test Precision: {results['test_precision']:.4f}")
    print(f"  Test Recall:    {results['test_recall']:.4f}")
    print(f"  Test F1 Score:  {results['test_f1']:.4f}")
    print(f"  Test AUC:       {results['test_auc']:.4f}")
    
    return model, results, y_pred_test, y_proba_test


def save_results(results_list, output_dir: str = "models"):
    """Save results to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results_file = f"{output_dir}/baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


def save_model_comparison(results_list):
    """Print model comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    comparison_df = pd.DataFrame(results_list)
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_idx = comparison_df['test_auc'].idxmax()
    best_model = comparison_df.loc[best_idx, 'model']
    best_auc = comparison_df.loc[best_idx, 'test_auc']
    
    print(f"\nBest model (by AUC): {best_model} ({best_auc:.4f})")


def train_models(csv_path: str = "data/processed/features.csv"):
    """Train and compare all baseline models."""
    print("\n" + "=" * 70)
    print("BASELINE MODEL TRAINING")
    print("=" * 70 + "\n")
    
    # Prepare data
    X_train, X_test, y_train, y_test, numeric_cols = prepare_data(csv_path)
    
    results_list = []
    
    # Train models
    try:
        model_lr, scaler_lr, results_lr, y_pred_lr, y_proba_lr = train_logistic_regression(
            X_train, X_test, y_train, y_test
        )
        results_list.append(results_lr)
    except Exception as e:
        print(f"Logistic Regression failed: {e}")
    
    try:
        model_xgb, results_xgb, y_pred_xgb, y_proba_xgb = train_xgboost(
            X_train, X_test, y_train, y_test
        )
        results_list.append(results_xgb)
    except Exception as e:
        print(f"XGBoost failed: {e}")
    
    # Compare and save
    if results_list:
        save_model_comparison(results_list)
        save_results(results_list)
    
    print("\n" + "=" * 70)
    print("Model training complete!")
    print("=" * 70)
    
    return results_list


if __name__ == "__main__":
    results = train_models()
