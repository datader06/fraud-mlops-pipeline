import pandas as pd
import yaml
import os


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def ingest_data():
    config = load_config()
    print(config)

    # Load raw data
    train_trans = pd.read_csv(config["data"]["raw_transaction_path"])
    train_id = pd.read_csv(config["data"]["raw_identity_path"])

    print("Transaction shape:", train_trans.shape)
    print("Identity shape:", train_id.shape)

    # Merge
    df = train_trans.merge(train_id, on="TransactionID", how="left")

    print("Merged shape:", df.shape)

    # Create output folder if not exists
    os.makedirs("data/processed", exist_ok=True)

    # Save parquet
    df.to_parquet(config["data"]["merged_path"], index=False)

    print(" Data saved at:", config["data"]["merged_path"])


if __name__ == "__main__":
    ingest_data()