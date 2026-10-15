from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

TARGET = "Machine failure"
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLS = ["UDI", "Product ID"]

def load_dataset(path: Path = RAW) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download AI4I 2020 CSV from "
            "https://archive.ics.uci.edu/dataset/601/ai4i+2020+ and save it as data/raw/ai4i2020.csv"
        )
    df = pd.read_csv(path)
    # Normalize whitespace in headers only; retain official column names otherwise.
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column {TARGET!r}. Found columns: {list(df.columns)}")
    return df

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    # Convert numeric-looking fields safely; leave Product ID and Type categorical.
    for col in df.columns:
        if col not in ["Product ID", "Type"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Derived features: use explicit formulas; speed is rpm and torque is Nm.
    if {"Air temperature [K]", "Process temperature [K]"}.issubset(df.columns):
        df["Temperature difference [K]"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
        )
    if {"Torque [Nm]", "Rotational speed [rpm]"}.issubset(df.columns):
        # Proportional power feature. rpm is converted to rad/s for physical power in watts.
        df["Estimated mechanical power [W]"] = (
            df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
        )
    return df

def model_features(df: pd.DataFrame):
    excluded = set(ID_COLS + FAILURE_MODE_COLS + [TARGET])
    features = [c for c in df.columns if c not in excluded]
    # Drop a categorical ID-like field from features; use Type as the only categorical predictor.
    if "Type" in features:
        categorical = ["Type"]
    else:
        categorical = []
    numeric = [c for c in features if c not in categorical]
    return features, numeric, categorical

def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    raw = load_dataset()
    print("Raw shape:", raw.shape)
    print("\nMissing values:\n", raw.isna().sum().sort_values(ascending=False).head(10))
    print("Exact duplicate rows:", int(raw.duplicated().sum()))
    cleaned = clean_and_engineer(raw)
    cleaned.to_csv(PROCESSED / "ai4i_cleaned.csv", index=False)
    cleaned.to_parquet(PROCESSED / "ai4i_cleaned.parquet", index=False)
    print("\nCleaned shape:", cleaned.shape)
    print("\nTarget counts:\n", cleaned[TARGET].value_counts(dropna=False))
    print("\nSaved CSV and Parquet to", PROCESSED)

if __name__ == "__main__":
    main()
