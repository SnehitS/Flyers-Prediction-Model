#!/usr/bin/env python3
"""Rebuild features.csv for 2024-25 and 2025-26 regular seasons."""
import sys
sys.path.insert(0, "src")

from build_dataset import SEASON_WINDOWS, build_dataset
import api_cache

start = SEASON_WINDOWS[0][0]
end = SEASON_WINDOWS[-1][1]
output_path = "data/processed/features.csv"

print(f"Building regular-season dataset from {start} to {end}...")
print("Schedule is cached under data/raw/cache/ (weekly NHL API calls).\n")

df = build_dataset(start, end, output_path)

if df.empty:
    print("FAILED - no games in dataset")
    sys.exit(1)

print(f"\n{'=' * 60}")
print(f"SUCCESS: {len(df)} regular-season games")
print(f"{'=' * 60}")
print(f"  File: {output_path}")
print(f"  Shape: {df.shape}")
print(f"  Dates: {df['gameDate'].min()} to {df['gameDate'].max()}")
print(f"  Home win rate: {df['home_win'].mean():.1%}")
print(f"  Cache: {api_cache.stats()}")

print("\nTeam stats populated:")
for col in ["home_wins", "home_losses", "away_wins", "away_losses", "home_rest_days", "away_rest_days"]:
    filled = df[col].notna().sum()
    print(f"  {col}: {filled}/{len(df)} ({100.0 * filled / len(df):.1f}%)")

print("\nRest-days summary:")
print(df[["home_rest_days", "away_rest_days"]].describe().to_string())

print("\nMissing values (nonzero only):")
missing = df.isnull().sum().sort_values(ascending=False)
for col, count in missing.items():
    if count > 0:
        print(f"  {col}: {count} ({100.0 * count / len(df):.1f}%)")
