#!/usr/bin/env python3
"""Quick test: rebuild just 3 days of games."""
import sys
sys.path.insert(0, 'src')

from datetime import date
from build_dataset import build_dataset

# Test with just 3 days
start = date(2025, 9, 20)
end = date(2025, 9, 22)

print(f"Testing rebuild with {start} to {end}...")
df = build_dataset(start, end, "data/processed/test_rebuild.csv")

if not df.empty:
    print(f"\nSUCCESS: Got {len(df)} games")
    print(f"Home win rate: {df['home_win'].mean():.1%}")
    
    print(f"\nTeam stats check:")
    for col in ['home_wins', 'home_losses', 'away_wins', 'away_losses']:
        filled = df[col].notna().sum()
        pct = 100.0 * filled / len(df)
        print(f"  {col}: {filled}/{len(df)} ({pct:.1f}%)")
        
    if df['home_wins'].notna().sum() > 0:
        print("\nOK: Team stats ARE populating!")
        # Show a few rows
        print("\nSample rows:")
        print(df[['gamePk', 'gameDate', 'home_win', 'home_wins', 'away_wins']].head(5))
    else:
        print("\nERROR: Team stats still NaN - fixes not working")
else:
    print("ERROR: No games found")
