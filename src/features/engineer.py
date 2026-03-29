import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT_DIR / "configs" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def build_features():
    config = load_config()

    print("Loading merged data...")

    df = pd.read_parquet(ROOT_DIR / config["data"]["merged_path"])

    print("Initial shape:", df.shape)

    # -------------------------
    # 1. TARGET
    # -------------------------
    y = df["isFraud"]
    X = df.drop(columns=["isFraud", "TransactionID"])

    # -------------------------
    # 2. TIME FEATURES
    # -------------------------
    X["hour"] = (X["TransactionDT"] // 3600) % 24
    X["day"] = (X["TransactionDT"] // (3600 * 24)) % 7

    # -------------------------
    # 3. LOG TRANSFORM
    # -------------------------
    X["TransactionAmt_log"] = np.log1p(X["TransactionAmt"])

    # -------------------------
    # 4. AGG FEATURES
    # -------------------------
    grp = X.groupby("card1")["TransactionAmt"]

    X["card1_mean"] = X["card1"].map(grp.mean())
    X["card1_std"] = X["card1"].map(grp.std())

    # -------------------------
    # 5. FREQUENCY ENCODING
    # -------------------------
    for col in ["card1", "card2", "P_emaildomain"]:
        freq = X[col].value_counts()
        X[col + "_freq"] = X[col].map(freq)

    # -------------------------
    # 6. HANDLE INF
    # -------------------------
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # -------------------------
    # 7. FILL NA
    # -------------------------
    X = X.fillna(-999)

    # -------------------------
    # 8. CATEGORICAL FIX (CRITICAL)
    # -------------------------
    cat_cols = X.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        X[col] = X[col].astype(str)
        X[col] = X[col].astype("category")

    print("Final shape:", X.shape)

    # -------------------------
    # 9. SAVE
    # -------------------------
    os.makedirs(ROOT_DIR / "data/processed", exist_ok=True)

    X["isFraud"] = y

    X.to_parquet(ROOT_DIR / config["data"]["processed_path"], index=False)

    print("Features saved at:", config["data"]["processed_path"])


if __name__ == "__main__":
    build_features()