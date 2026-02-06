#!/usr/bin/env python3
"""Run complete EDA and baseline model training pipeline."""

import sys
sys.path.insert(0, 'src')

from eda import run_eda
from train_model import train_models

print("=" * 70)
print("FLYERS PREDICTION MODEL - COMPLETE PIPELINE")
print("=" * 70)

# Step 1: EDA
print("\nStep 1: Running Exploratory Data Analysis...")
df = run_eda()

# Step 2: Baseline Models
print("\nStep 2: Training Baseline Models...")
results = train_models()

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("  - Review EDA plots in data/processed/eda_plots/")
print("  - Check baseline results in models/baseline_results.json")
print("  - Consider feature engineering to improve model performance")
