"""Train baseline models for NHL home_win prediction.

Uses a chronological split: train through Feb 2026, validate March–April 2026
so future games never leak into training.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Outcome / identity columns must not be features.
EXCLUDE_COLS = {
    "gamePk",
    "gameDate",
    "home_win",
    "away_win",
    "result_type",
    "period_type",
    "home_score",
    "away_score",
    "home_id",
    "away_id",
    "home_abbrev",
    "away_abbrev",
    "season",
}

# Last day included in training (2025-26 regular season continues after this).
TRAIN_END = pd.Timestamp("2026-02-28")


def _metrics(name: str, y_train, y_pred_train, y_test, y_pred_test, y_proba_test) -> dict:
    return {
        "model": name,
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "test_precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "test_auc": float(roc_auc_score(y_test, y_proba_test)),
    }


def _print_results(results: dict) -> None:
    print("\nResults:")
    print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
    print(f"  Test Precision: {results['test_precision']:.4f}")
    print(f"  Test Recall:    {results['test_recall']:.4f}")
    print(f"  Test F1 Score:  {results['test_f1']:.4f}")
    print(f"  Test AUC:       {results['test_auc']:.4f}")


def prepare_data(
    csv_path: str = "data/processed/features.csv",
    train_end: pd.Timestamp = TRAIN_END,
):
    """Load features and split by game date (no shuffle)."""
    print("Loading data...")
    df = pd.read_csv(csv_path, parse_dates=["gameDate"])
    df = df.sort_values(["gameDate", "gamePk"]).reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['gameDate'].min().date()} to {df['gameDate'].max().date()}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in EXCLUDE_COLS
    ]
    print(f"Features selected ({len(numeric_cols)}): {numeric_cols}")

    df = df[df["home_win"].notna()].copy()
    train_mask = df["gameDate"] <= train_end
    test_mask = df["gameDate"] > train_end

    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Chronological split on {train_end.date()} produced "
            f"train={len(train_df)} test={len(test_df)}. Check gameDate coverage."
        )

    X_train = train_df[numeric_cols].copy()
    X_test = test_df[numeric_cols].copy()
    y_train = train_df["home_win"].astype(int)
    y_test = test_df["home_win"].astype(int)

    fill_values = X_train.median(numeric_only=True).fillna(0)
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)

    naive_acc = float(y_test.mean())
    print(f"Train set: {len(X_train)} samples ({train_df['gameDate'].min().date()} to {train_df['gameDate'].max().date()})")
    print(f"Test set:  {len(X_test)} samples ({test_df['gameDate'].min().date()} to {test_df['gameDate'].max().date()})")
    print(f"Home win rate (train): {y_train.mean():.1%}")
    print(f"Home win rate (test):  {y_test.mean():.1%}  (always-home-win accuracy)")
    print(f"Missing values after filling: {int(X_train.isnull().sum().sum() + X_test.isnull().sum().sum())}")

    return X_train, X_test, y_train, y_test, numeric_cols, fill_values, naive_acc


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train logistic regression baseline."""
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training logistic regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    results = _metrics(
        "Logistic Regression", y_train, y_pred_train, y_test, y_pred_test, y_proba_test
    )
    _print_results(results)

    coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
    print("\nTop coefficients (signed):")
    for name, value in coefs.head(10).items():
        print(f"  {name:30s}: {value:+.4f}")

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
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    results = _metrics("XGBoost", y_train, y_pred_train, y_test, y_pred_test, y_proba_test)
    _print_results(results)

    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop features:")
    for name, value in importance.head(10).items():
        print(f"  {name:30s}: {value:.4f}")

    return model, results, y_pred_test, y_proba_test


def save_results(results_list, output_dir: str = "models"):
    """Save metrics JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_file = Path(output_dir) / "baseline_results.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)
    print(f"\nResults saved to {results_file}")


def save_artifacts(payload: dict, output_dir: str = "models") -> None:
    """Persist the best model plus the feature list used to train it."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / "home_win_model.joblib"
    joblib.dump(payload, path)
    print(f"Model saved to {path}")


def save_model_comparison(results_list, naive_acc: float):
    """Print model comparison table against the always-home-win baseline."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    rows = list(results_list) + [
        {
            "model": "Always home win",
            "train_accuracy": None,
            "test_accuracy": naive_acc,
            "test_precision": naive_acc,
            "test_recall": 1.0,
            "test_f1": None,
            "test_auc": 0.5,
        }
    ]
    comparison_df = pd.DataFrame(rows)
    print(comparison_df.to_string(index=False))

    scored = comparison_df[comparison_df["model"] != "Always home win"]
    best_idx = scored["test_auc"].idxmax()
    best_model = scored.loc[best_idx, "model"]
    best_auc = scored.loc[best_idx, "test_auc"]
    print(f"\nBest model (by AUC): {best_model} ({best_auc:.4f})")
    print(f"Always-home-win test accuracy: {naive_acc:.4f}")


def train_models(csv_path: str = "data/processed/features.csv"):
    """Train and compare baseline models on a chronological holdout."""
    print("\n" + "=" * 70)
    print("BASELINE MODEL TRAINING (chronological split)")
    print("=" * 70 + "\n")

    X_train, X_test, y_train, y_test, numeric_cols, fill_values, naive_acc = prepare_data(csv_path)

    results_list = []
    artifacts = {
        "features": numeric_cols,
        "fill_values": fill_values,
        "train_end": str(TRAIN_END.date()),
        "naive_home_win_rate": naive_acc,
    }

    try:
        model_lr, scaler_lr, results_lr, _, _ = train_logistic_regression(
            X_train, X_test, y_train, y_test
        )
        results_list.append(results_lr)
        artifacts["logistic_regression"] = model_lr
        artifacts["scaler"] = scaler_lr
    except Exception as e:
        print(f"Logistic Regression failed: {e}")

    try:
        model_xgb, results_xgb, _, _ = train_xgboost(X_train, X_test, y_train, y_test)
        results_list.append(results_xgb)
        artifacts["xgboost"] = model_xgb
    except Exception as e:
        print(f"XGBoost failed: {e}")

    if results_list:
        save_model_comparison(results_list, naive_acc)
        save_results(results_list)
        best_name = max(results_list, key=lambda r: r["test_auc"])["model"]
        artifacts["best_model"] = best_name
        save_artifacts(artifacts)

    print("\n" + "=" * 70)
    print("Model training complete!")
    print("=" * 70)
    return results_list


if __name__ == "__main__":
    train_models()
