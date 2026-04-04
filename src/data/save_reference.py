import pandas as pd
import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT_DIR / "configs" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def save_reference():
    config = load_config()

    df = pd.read_parquet(ROOT_DIR / config["data"]["processed_path"])

    # Use the first 70% as stable historical reference
    split_idx = int(len(df) * 0.7)
    reference_df = df.iloc[:split_idx].copy()

    reference_path = ROOT_DIR / config["data"]["reference_path"]
    reference_path.parent.mkdir(parents=True, exist_ok=True)

    reference_df.to_parquet(reference_path, index=False)

    print(f"Reference dataset saved at: {reference_path}")
    print(f"Reference shape: {reference_df.shape}")


if __name__ == "__main__":
    save_reference()