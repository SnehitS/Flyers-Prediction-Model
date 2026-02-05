"""Quick build of recent games to test everything works."""

import sys
sys.path.insert(0, 'src')

from datetime import date
from build_dataset import build_dataset

# Build just recent games first (Jan 2025 - Feb 2026) to verify
start = date(2025, 1, 1)
end = date(2026, 2, 4)

print(f"Building dataset from {start} to {end}...")
df = build_dataset(start, end, "data/processed/features.csv")

if not df.empty:
    print(f"\nSUCCESS!")
    print(f"  Shape: {df.shape}")
    print(f"  Home win rate: {df['home_win'].mean():.1%}")
    print(f"  First few rows:")
    print(df.head())
else:
    print(f"\nFAILED - no games in dataset")
