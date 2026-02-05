import pandas as pd

path = "data/processed/features.csv"
print(f"Loading: {path}")
try:
    df = pd.read_csv(path)
except Exception as e:
    print(f"FAILED to load CSV: {e}")
    raise

print('\nShape: ', df.shape)
print('\nColumns:')
print(df.columns.tolist())

# Label distribution
if 'home_win' in df.columns:
    print('\nhome_win distribution:')
    print(df['home_win'].value_counts(dropna=False))
else:
    print('\nNo home_win column found')

# Missingness summary
missing_counts = df.isnull().sum().sort_values(ascending=False)
missing_frac = (df.isnull().mean() * 100).sort_values(ascending=False)
print('\nTop 30 missing counts:')
print(missing_counts.head(30))
print('\nTop 30 missing percent:')
print(missing_frac.head(30))

# Show rows with missing label
if 'home_win' in df.columns:
    null_label = df[df['home_win'].isnull()]
    print(f"\nRows with missing home_win: {len(null_label)}")

# Sample rows
print('\nSample rows (first 5):')
print(df.head().to_string())

# Quick completeness check: fraction of rows with any nulls
any_null = df.isnull().any(axis=1).mean()
print(f"\nFraction of rows with any missing value: {any_null:.3%}")
