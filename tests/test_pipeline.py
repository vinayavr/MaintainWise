import pandas as pd
from src.data_pipeline import clean_and_engineer, model_features, TARGET

def test_feature_engineering_and_leakage_exclusion():
    sample = pd.DataFrame({
        "UDI": [1, 2],
        "Product ID": ["M1", "M2"],
        "Type": ["M", "L"],
        "Air temperature [K]": [300.0, 301.0],
        "Process temperature [K]": [310.0, 311.0],
        "Rotational speed [rpm]": [1500, 1600],
        "Torque [Nm]": [40.0, 35.0],
        "Tool wear [min]": [100, 120],
        "Machine failure": [0, 1],
        "TWF": [0, 1], "HDF": [0, 0], "PWF": [0, 0], "OSF": [0, 0], "RNF": [0, 0]
    })
    clean = clean_and_engineer(sample)
    assert "Temperature difference [K]" in clean.columns
    assert "Estimated mechanical power [W]" in clean.columns
    features, _, _ = model_features(clean)
    assert TARGET not in features
    assert not any(col in features for col in ["TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"])
