"""Exploratory Data Analysis for NHL home_win prediction dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_dataset(csv_path: str = "data/processed/features.csv") -> pd.DataFrame:
    """Load features CSV."""
    return pd.read_csv(csv_path)


def check_data_quality(df: pd.DataFrame) -> None:
    """Print data quality summary."""
    print("=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")
    
    print(f"\n--- Missing Values (Top 15) ---")
    missing = df.isnull().sum().sort_values(ascending=False)
    for col, count in missing.head(15).items():
        pct = 100.0 * count / len(df)
        if count > 0:
            print(f"  {col:30s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n--- Label Distribution ---")
    if 'home_win' in df.columns:
        print(f"  home_win value counts:")
        print(df['home_win'].value_counts().sort_index())
        print(f"  Home win rate: {df['home_win'].mean():.1%}")
    
    print(f"\n--- Numeric Features Summary ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_df = df[numeric_cols].describe().T
    print(stats_df[['count', 'mean', 'std', 'min', 'max']].to_string())


def check_team_stats(df: pd.DataFrame) -> None:
    """Verify team stats are populated."""
    print("\n" + "=" * 70)
    print("TEAM STATS POPULATION CHECK")
    print("=" * 70)
    
    team_cols = ['home_wins', 'home_losses', 'away_wins', 'away_losses']
    
    for col in team_cols:
        if col in df.columns:
            filled = df[col].notna().sum()
            pct = 100.0 * filled / len(df)
            print(f"  {col:20s}: {filled:3d}/{len(df)} ({pct:5.1f}%)")
        else:
            print(f"  {col:20s}: MISSING")


def analyze_features(df: pd.DataFrame) -> None:
    """Analyze feature distributions and correlations."""
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)
    
    # Numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'home_win' in numeric_cols:
        numeric_cols.remove('home_win')
    
    print(f"\nNumeric features ({len(numeric_cols)}):")
    for col in numeric_cols[:10]:
        print(f"  - {col}")
    if len(numeric_cols) > 10:
        print(f"  ... and {len(numeric_cols) - 10} more")
    
    # Correlation with target
    if 'home_win' in df.columns:
        print(f"\n--- Correlation with home_win (Top 10) ---")
        correlations = df[numeric_cols + ['home_win']].corr()['home_win'].drop('home_win').abs().sort_values(ascending=False)
        for col, corr in correlations.head(10).items():
            print(f"  {col:30s}: {corr:+.3f}")


def save_visualizations(df: pd.DataFrame, output_dir: str = "data/processed/eda_plots") -> None:
    """Save exploratory plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving plots to {output_dir}/")
    
    # Label distribution
    if 'home_win' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        df['home_win'].value_counts().sort_index().plot(kind='bar', ax=ax)
        ax.set_title('Home Win Label Distribution')
        ax.set_xlabel('Home Win (0=Loss, 1=Win)')
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/label_distribution.png", dpi=100)
        plt.close()
        print(f"  ✓ label_distribution.png")
    
    # Missing values
    missing = df.isnull().sum().sort_values(ascending=False).head(15)
    if missing.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing[missing > 0].plot(kind='barh', ax=ax)
        ax.set_title('Missing Values (Top 15)')
        ax.set_xlabel('Count')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/missing_values.png", dpi=100)
        plt.close()
        print(f"  ✓ missing_values.png")
    
    # Feature distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'home_win' in numeric_cols:
        numeric_cols.remove('home_win')
    
    if numeric_cols:
        # Select top 9 features for visualization
        cols_to_plot = numeric_cols[:9]
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(cols_to_plot):
            data = df[col].dropna()
            if len(data) > 0:
                axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(col)
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_distributions.png", dpi=100)
        plt.close()
        print(f"  ✓ feature_distributions.png")


def run_eda(csv_path: str = "data/processed/features.csv") -> pd.DataFrame:
    """Run complete EDA."""
    print("\nLoading dataset...")
    df = load_dataset(csv_path)
    
    check_data_quality(df)
    check_team_stats(df)
    analyze_features(df)
    save_visualizations(df)
    
    print("\n" + "=" * 70)
    print("EDA Complete!")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    df = run_eda()
