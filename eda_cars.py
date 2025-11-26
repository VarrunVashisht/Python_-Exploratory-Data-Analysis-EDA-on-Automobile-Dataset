# eda_cars.py  — PyCharm friendly (safe + full charts)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

# 1. Load data
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
df = pd.read_csv(path)
print("Shape:", df.shape)
print(df.head())

# 2. Replace missing placeholder and coerce numeric columns
df.replace("?", np.nan, inplace=True)

# define numeric columns we expect to have numbers
numeric_cols = [
    "symboling", "normalized-losses", "wheel-base", "length", "width", "height",
    "curb-weight", "engine-size", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
]
numeric_cols = [c for c in numeric_cols if c in df.columns]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")   # invalid -> NaN

print("\nData types (after conversion):\n", df.dtypes)

# 3. Correlation matrix (numeric only) — safe printing
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()
print("\nCorrelation matrix (numeric columns only):\n", corr)

# 4. Regression plot: engine-size vs price
# NOTE: if NaNs remain, seaborn will ignore them automatically
plt.figure(figsize=(6,4))
sns.regplot(x="engine-size", y="price", data=df, scatter_kws={'alpha':0.6})
plt.ylim(0,)
plt.title("Engine size vs Price (regression)")
plt.xlabel("Engine size")
plt.ylabel("Price")
plt.show()

# 5. Regression plot: highway-mpg vs price
plt.figure(figsize=(6,4))
sns.regplot(x="highway-mpg", y="price", data=df, scatter_kws={'alpha':0.6})
plt.title("Highway MPG vs Price (regression)")
plt.xlabel("Highway MPG")
plt.ylabel("Price")
plt.show()

# 6. Regression plot: peak-rpm vs price
plt.figure(figsize=(6,4))
sns.regplot(x="peak-rpm", y="price", data=df, scatter_kws={'alpha':0.6})
plt.title("Peak RPM vs Price (regression)")
plt.xlabel("Peak RPM")
plt.ylabel("Price")
plt.show()

# 7. Boxplot: price by body-style
plt.figure(figsize=(8,5))
sns.boxplot(x="body-style", y="price", data=df)
plt.title("Price distribution by body style")
plt.xticks(rotation=30)
plt.xlabel("Body Style")
plt.ylabel("Price")
plt.show()

# 8. Boxplot: price by engine-location
plt.figure(figsize=(6,4))
sns.boxplot(x="engine-location", y="price", data=df)
plt.title("Price by engine location")
plt.xlabel("Engine Location")
plt.ylabel("Price")
plt.show()

# 9. Numeric and object summaries
print("\nNumeric summary:\n", df[numeric_cols].describe().transpose())
print("\nObject summary (categorical columns):\n", df.describe(include=['object']))

# 10. Value counts example for drive-wheels
if 'drive-wheels' in df.columns:
    drive_counts = df['drive-wheels'].value_counts().to_frame().rename(columns={'drive-wheels':'value_counts'})
    drive_counts.index.name = 'drive-wheels'
    print("\nDrive wheels counts:\n", drive_counts)

# 11. Grouping and pivot (drive-wheels x body-style average price)
if set(['drive-wheels','body-style','price']).issubset(df.columns):
    grouped = df[['drive-wheels','body-style','price']].groupby(['drive-wheels','body-style'], as_index=False).mean()
    pivot = grouped.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)
    print("\nPivot table (avg price):\n", pivot)

    # 12. Pcolor heatmap (average price)
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap='RdBu', linewidths=0.5)
    plt.title("Average price: drive-wheels x body-style (heatmap)")
    plt.xlabel("Body Style")
    plt.ylabel("Drive Wheels")
    plt.show()

# 13. Optional pairplot for a few key numeric columns (commented out for speed)
# Use this only if you want pairwise scatter + histograms (can be slow)
# small_list = ['price','engine-size','horsepower','curb-weight']
# sns.pairplot(df[small_list].dropna(), kind='scatter', diag_kind='kde')
# plt.show()

# 14. Pearson correlation coefficients with p-values for selected pairs
pairs = ['wheel-base','horsepower','length','width','curb-weight','engine-size','bore','city-mpg','highway-mpg']
print("\nPearson correlations with price (coef, p-value):")
for col in pairs:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        # drop rows where either is NaN
        subset = df[[col, 'price']].dropna()
        if len(subset) > 2:   # pearson requires at least 3 observations
            pearson_coef, p_value = stats.pearsonr(subset[col], subset['price'])
            print(f"{col:12s} -> Pearson r = {pearson_coef:.3f}, p-value = {p_value:.5g}")
        else:
            print(f"{col:12s} -> Not enough data to compute Pearson")

# 15. Histogram of price (distribution)
plt.figure(figsize=(7,4))
plt.hist(df['price'].dropna(), bins=20, edgecolor='black')
plt.title("Price distribution (histogram)")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# 16. KDE for price (smoothed density)
plt.figure(figsize=(7,4))
sns.kdeplot(df['price'].dropna(), shade=True)
plt.title("Price density (KDE)")
plt.xlabel("Price")
plt.show()

print("\nAll plots complete.")
