"""
Phase 3 utility — Save Reference Dataset for Drift Monitoring

Extracts the first N rows of the training feature set as the 'reference'
baseline that Evidently will compare incoming batches against in Phase 6.

Run from the project root:
    python -m src.data.save_reference
"""
import pandas as pd
import yaml
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT_DIR / "configs" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def save_reference(n_rows: int = 10_000):
    """
    Save the first `n_rows` rows of the training feature set as the
    reference baseline for drift detection.

    The reference set should represent 'known-good' production-like traffic.
    Using the first slice (earliest transactions) simulates the training window.
    """
    config = load_config()

    features_path   = ROOT_DIR / config["data"]["processed_path"]
    reference_path  = ROOT_DIR / config["data"]["reference_path"]

    print(f"Loading features from: {features_path}")
    df = pd.read_parquet(features_path)

    # Drop the target — drift monitor only sees features
    if "isFraud" in df.columns:
        df = df.drop(columns=["isFraud"])

    reference = df.head(n_rows).copy()

    os.makedirs(reference_path.parent, exist_ok=True)
    reference.to_parquet(reference_path, index=False)

    print(f"✅ Reference dataset saved → {reference_path}")
    print(f"   Shape: {reference.shape}")
    print(f"   This slice will be used as the baseline for Evidently drift detection.")


if __name__ == "__main__":
    save_reference()
