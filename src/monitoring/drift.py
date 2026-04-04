import pandas as pd
import yaml
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT_DIR / "configs" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_drift_check():
    config = load_config()

    reference_path = ROOT_DIR / config["data"]["reference_path"]
    processed_path = ROOT_DIR / config["data"]["processed_path"]

    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(processed_path)

    # Simulate current batch as the most recent 30% of data
    split_idx = int(len(current_df) * 0.7)
    current_df = current_df.iloc[split_idx:].copy()

    # Drop target if present for feature drift check
    if "isFraud" in reference_df.columns:
        reference_features = reference_df.drop(columns=["isFraud"])
    else:
        reference_features = reference_df.copy()

    if "isFraud" in current_df.columns:
        current_features = current_df.drop(columns=["isFraud"])
    else:
        current_features = current_df.copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_features, current_data=current_features)

    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    output_path = reports_dir / "drift_report.html"
    report.save_html(str(output_path))

    print(f"Drift report saved at: {output_path}")


if __name__ == "__main__":
    run_drift_check()