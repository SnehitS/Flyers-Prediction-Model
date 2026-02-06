#!/usr/bin/env python3
"""Test rebuild with just one day of games to verify pipeline."""
import sys
sys.path.insert(0, 'src')

from datetime import date
from build_dataset import build_dataset

print("Quick test: building 1 day (2025-09-20)...")
df = build_dataset(date(2025, 9, 20), date(2025, 9, 20), "data/processed/test_one_day.csv")

if not df.empty:
    print(f"\nSUCCESS: Got {len(df)} games")
    print(f"Home win rate: {df['home_win'].mean():.1%}")
    
    print(f"\nTeam stats check:")
    for col in ['home_wins', 'home_losses', 'away_wins', 'away_losses']:
        filled = df[col].notna().sum()
        pct = 100.0 * filled / len(df)
        print(f"  {col}: {filled}/{len(df)} ({pct:.1f}%)")
    
    if df['home_wins'].notna().sum() > 0:
        print("\n✓ Team stats ARE populating correctly!")
    else:
        print("\n✗ Team stats NOT populating - still seeing NaNs")
else:
    print("No games found for this date")
