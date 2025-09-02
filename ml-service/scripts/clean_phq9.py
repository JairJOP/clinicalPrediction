# ml-service/scripts/clean_phq9.py
# Preprocess PHQ-9 Student dataset following the proposal:
# - Map PHQ-9 Likert text -> numeric 0..3
# - Handle missing values (mean for numeric, most-frequent for categorical)
# - Normalise bounded features with min-max; continuous with z-score
# - One-hot encode categorical (e.g., gender)
# - Create binary target via PHQ-9 >= 10
# - Save processed CSVs and a stratified train/test split

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------
# Paths
# --------------------
ROOT = Path(__file__).parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Updated_PHQ9_Student_Dataset.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# Config
# --------------------
LIKERT_MAP = {
    "not at all": 0,
    "several days": 1,
    "more than half the days": 2,
    "nearly every day": 3,
}
PHQ_TOTAL_THRESHOLD = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --------------------
# Helpers
# --------------------
def normalise_colname(s: str) -> str:
    s = (s or "").strip().lower()
    # we replace non-alphanumeric with underscores
    s = re.sub(r"[^0-9a-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def is_likert_series(ser: pd.Series) -> bool:
    """Detect PHQ-9 Likert columns by value set (case-insensitive)."""
    if ser.dtype.kind not in ("O", "U", "S"):
        return False
    vals = (
        pd.Series(ser.dropna().astype(str).str.strip().str.lower().unique())
        .dropna()
        .tolist()
    )
    if not vals:
        return False
    allowed = set(LIKERT_MAP.keys())
    return set(vals).issubset(allowed)

def minmax_scale(arr, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.nanmin(arr)
    if max_val is None:
        max_val = np.nanmax(arr)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return (arr - min_val) / denom

def zscore(arr, mean=None, std=None):
    if mean is None:
        mean = np.nanmean(arr)
    if std is None or std == 0:
        std = np.nanstd(arr)
        if std == 0:
            std = 1.0
    return (arr - mean) / std

# --------------------
# Load
# --------------------
print(f"Loading: {RAW_PATH}")
df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")

# --------------------
# Column normalisation
# --------------------
df.columns = [normalise_colname(c) for c in df.columns]

# We keep a copy of raw
raw_df = df.copy()

# --------------------
# Map PHQ-9 Likert -> numeric 0..3
# (detect columns whose values are subset of the four Likert options)
# --------------------
phq_item_cols = []
for col in df.columns:
    if is_likert_series(df[col]):
        phq_item_cols.append(col)
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(LIKERT_MAP)
            .astype("float")
        )

# If a precomputed total exists, we’ll recompute anyway for consistency
# (based on the 9 mapped items we found)
if len(phq_item_cols) == 9:
    df["phq9_total"] = df[phq_item_cols].sum(axis=1)
else:
    # Try to find an existing total column
    cand = [c for c in df.columns if "phq" in c and "total" in c]
    if cand:
        df["phq9_total"] = pd.to_numeric(df[cand[0]], errors="coerce")
    else:
        # If we can’t locate all 9 items, we’ll sum what we have (best effort)
        if phq_item_cols:
            df["phq9_total"] = df[phq_item_cols].sum(axis=1)
        else:
            raise ValueError("Could not detect PHQ-9 item columns in this file.")

# --------------------
# Basic typing & selection
# --------------------
# Identify candidate continuous columns (e.g., age if present)
continuous_cols = []
if "age" in df.columns:
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    continuous_cols.append("age")

# Identify categorical columns we’ll encode (e.g., gender)
categorical_cols = []
for cand in ["gender", "sex"]:
    if cand in df.columns:
        categorical_cols.append(cand)

# --------------------
# Missing value handling
# - Numeric: mean imputation
# - Categorical: most frequent (mode)
# --------------------
numeric_cols = list(set(phq_item_cols + continuous_cols + ["phq9_total"]))
for col in numeric_cols:
    if col in df.columns:
        mean_val = df[col].astype(float).mean(skipna=True)
        df[col] = df[col].fillna(mean_val)

for col in categorical_cols:
    mode_val = df[col].mode(dropna=True)
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val.iloc[0])

# --------------------
# Normalisation
# - Bounded (PHQ items 0..3): min-max (divide by 3)
# - PHQ-9 total (0..27): min-max (divide by 27)
# - Continuous (e.g., age): z-score
# --------------------
for col in phq_item_cols:
    df[col] = df[col] / 3.0  # 0..1 by design

if "phq9_total" in df.columns:
    df["phq9_total_norm"] = df["phq9_total"] / 27.0  # 0..1

for col in continuous_cols:
    df[col + "_z"] = zscore(df[col].astype(float).values)

# --------------------
# Categorical encoding
# - One-hot encode non-ordinal categoricals (drop_first=True)
# --------------------
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --------------------
# Target variable (binary): PHQ-9 total >= 10 => 1 (Depression), else 0
# --------------------
df["depression"] = (df["phq9_total"] >= PHQ_TOTAL_THRESHOLD).astype(int)

# --------------------
# --------------------
# Feature selection (updated)
# We derive the label from PHQ-9 TOTAL, but DO NOT include totals as features.
# --------------------
feature_cols = []

# 1) Use the 9 PHQ item columns (mapped 0..3 and already scaled /3 to 0..1)
feature_cols += phq_item_cols

# 2) Add continuous z-scored columns (e.g., age_z) – optional but fine
feature_cols += [c for c in df.columns if c.endswith("_z")]

# 3) Add one-hot encoded categoricals (e.g., gender_*, sex_*)
feature_cols += [c for c in df.columns if c.startswith("gender_") or c.startswith("sex_")]

# IMPORTANT: exclude any total/normalized totals from features
feature_cols = [c for c in feature_cols if c not in ("phq9_total", "phq9_total_norm")]

# Remove dups while preserving order
seen = set()
feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

X = df[feature_cols].copy()
y = df["depression"].copy()

# Safety
X = X.replace([np.inf, -np.inf], np.nan)
if X.isna().any().any():
    X = X.fillna(X.mean(numeric_only=True))

# --------------------
# Save processed full dataset
# --------------------
proc_path = OUT_DIR / "phq9_processed_full.csv"
df_out = X.copy()
df_out["depression"] = y
df_out.to_csv(proc_path, index=False)
print(f"💾 Saved processed dataset: {proc_path}")

# --------------------
# Stratified train/test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

train_path = OUT_DIR / "phq9_train.csv"
test_path = OUT_DIR / "phq9_test.csv"
pd.concat([X_train, y_train.rename("depression")], axis=1).to_csv(train_path, index=False)
pd.concat([X_test, y_test.rename("depression")], axis=1).to_csv(test_path, index=False)

print(f"Train/Test saved:\n - {train_path}\n - {test_path}")
print(f"Features used ({len(feature_cols)}): {feature_cols}")
