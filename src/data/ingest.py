import pandas as pd
import yaml
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT_DIR / "configs" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def ingest_data():
    config = load_config()
    print("Config loaded:", config)

    raw_transaction_path = ROOT_DIR / config["data"]["raw_transaction_path"]
    raw_identity_path = ROOT_DIR / config["data"]["raw_identity_path"]
    merged_path = ROOT_DIR / config["data"]["merged_path"]

    # Load raw data
    print(f"Loading transactions from: {raw_transaction_path}")
    train_trans = pd.read_csv(raw_transaction_path)

    print(f"Loading identity from: {raw_identity_path}")
    train_id = pd.read_csv(raw_identity_path)

    print("Transaction shape:", train_trans.shape)
    print("Identity shape:", train_id.shape)

    # Merge on TransactionID
    df = train_trans.merge(train_id, on="TransactionID", how="left")

    print("Merged shape:", df.shape)

    # Create output folder if not exists
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    df.to_parquet(merged_path, index=False)

    print(f"Data saved at: {merged_path}")


if __name__ == "__main__":
    ingest_data()