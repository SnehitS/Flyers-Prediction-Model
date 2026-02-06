#!/usr/bin/env python3
"""Start the rebuild in background and report progress."""
import subprocess
import sys
import time
import os

os.chdir(r'c:\Users\shada\flyers-prediction-model')

print("Starting rebuild_dataset.py in background...")
print("This will take several minutes (100+ API calls per game)\n")

# Start background process
proc = subprocess.Popen(
    [sys.executable, 'rebuild_dataset.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

print(f"Process ID: {proc.pid}")
print("Waiting for completion...\n")

# Wait for process
try:
    stdout, stderr = proc.communicate(timeout=900)  # 15 min timeout
    
    print(stdout)
    
    if proc.returncode == 0:
        print("\nSUCCESS: Dataset rebuild completed!")
        
        # Check the CSV
        import pandas as pd
        df = pd.read_csv('data/processed/features.csv')
        
        print(f"\nDataset shape: {df.shape}")
        print(f"Home win rate: {df['home_win'].mean():.1%}")
        
        print(f"\nTeam stats check:")
        for col in ['home_wins', 'home_losses', 'away_wins', 'away_losses']:
            filled = df[col].notna().sum()
            pct = 100.0 * filled / len(df)
            print(f"  {col}: {filled}/{len(df)} ({pct:.1f}%)")
    else:
        print(f"\nERROR: Process exited with code {proc.returncode}")
        
except subprocess.TimeoutExpired:
    proc.kill()
    print("ERROR: Process timed out after 15 minutes")
