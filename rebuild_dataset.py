#!/usr/bin/env python3
"""Rebuild features.csv with fixed API calls."""
import sys
sys.path.insert(0, 'src')

from datetime import date
from build_dataset import build_dataset
import os

# Build dataset
start = date(2025, 1, 1)
end = date(2026, 2, 4)

print(f"Building dataset from {start} to {end}...")
print(f"This will fetch all games and assemble features with team stats...\n")

output_path = "data/processed/features.csv"
df = build_dataset(start, end, output_path)

if not df.empty:
    print(f"\n{'='*60}")
    print(f"✓ SUCCESS! Dataset built with {len(df)} games")
    print(f"{'='*60}")
    print(f"\nDataset Statistics:")
    print(f"  File: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Home win rate: {df['home_win'].mean():.1%}")
    
    # Check team stats population
    print(f"\nTeam Stats Population:")
    for col in ['home_wins', 'home_losses', 'away_wins', 'away_losses']:
        if col in df.columns:
            filled = df[col].notna().sum()
            pct = 100.0 * filled / len(df)
            print(f"  {col}: {filled}/{len(df)} ({pct:.1f}%)")
    
    print(f"\nMissing values by column (top 10):")
    missing = df.isnull().sum().sort_values(ascending=False)
    for col, count in missing.head(10).items():
        pct = 100.0 * count / len(df)
        if count > 0:
            print(f"  {col}: {count:4d} ({pct:5.1f}%)")
else:
    print("\n✗ FAILED - no games in dataset")
    sys.exit(1)
