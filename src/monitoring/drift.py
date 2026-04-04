import subprocess
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT_DIR / "configs" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def calculate_psi(expected, actual, bins=10):
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()

    if expected.empty or actual.empty:
        return np.nan

    breakpoints = np.linspace(0, 100, bins + 1)
    quantiles = np.percentile(expected, breakpoints)

    # avoid duplicate bin edges
    quantiles = np.unique(quantiles)
    if len(quantiles) < 2:
        return 0.0

    expected_bins = pd.cut(expected, bins=quantiles, include_lowest=True)
    actual_bins = pd.cut(actual, bins=quantiles, include_lowest=True)

    expected_dist = expected_bins.value_counts(normalize=True).sort_index()
    actual_dist = actual_bins.value_counts(normalize=True).sort_index()

    all_idx = expected_dist.index.union(actual_dist.index)
    expected_dist = expected_dist.reindex(all_idx, fill_value=1e-6)
    actual_dist = actual_dist.reindex(all_idx, fill_value=1e-6)

    psi = ((expected_dist - actual_dist) * np.log(expected_dist / actual_dist)).sum()
    return float(psi)


def trigger_retraining():
    print("\n Triggering retraining pipeline...\n")

    result = subprocess.run(
        ["python", "src/models/train.py"],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR
    )

    print("Retraining output:\n")
    print(result.stdout)

    if result.returncode != 0:
        print(" Retraining failed!")
        print(result.stderr)
    else:
        print(" Retraining completed successfully!")


def run_drift_check():
    config = load_config()

    reference_path = ROOT_DIR / config["data"]["reference_path"]
    processed_path = ROOT_DIR / config["data"]["processed_path"]

    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(processed_path)

    # simulate latest batch as most recent 30%
    split_idx = int(len(current_df) * 0.7)
    current_df = current_df.iloc[split_idx:].copy()

    # split target from features
    if "isFraud" in reference_df.columns:
        reference_features = reference_df.drop(columns=["isFraud"])
        reference_target = reference_df["isFraud"]
    else:
        reference_features = reference_df.copy()
        reference_target = None

    if "isFraud" in current_df.columns:
        current_features = current_df.drop(columns=["isFraud"])
        current_target = current_df["isFraud"]
    else:
        current_features = current_df.copy()
        current_target = None

    # Evidently HTML report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_features, current_data=current_features)

    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    output_path = reports_dir / "drift_report.html"
    report.save_html(str(output_path))

    print(f" Drift report saved at: {output_path}")

    # PSI check on numeric columns
    psi_threshold = config["drift"]["psi_threshold"]
    top_n = config["drift"].get("top_n_features", 20)

    numeric_cols = reference_features.select_dtypes(include=["number"]).columns.tolist()
    drifted_features = []

    print("\n PSI results:")
    for col in numeric_cols[:top_n]:
        psi = calculate_psi(reference_features[col], current_features[col])
        if pd.notna(psi):
            print(f"{col}: PSI={psi:.4f}")
            if psi > psi_threshold:
                drifted_features.append((col, psi))

    # optional label drift check
    label_drift_detected = False
    if reference_target is not None and current_target is not None:
        ref_rate = reference_target.mean()
        cur_rate = current_target.mean()
        label_shift = abs(cur_rate - ref_rate)

        print("\n Label drift check:")
        print(f"Reference fraud rate: {ref_rate:.4f}")
        print(f"Current fraud rate:   {cur_rate:.4f}")
        print(f"Absolute shift:       {label_shift:.4f}")

        if label_shift > 0.02:
            label_drift_detected = True

    # summary + retraining trigger
    print("\n Drift summary:")
    if drifted_features:
        print("Significant feature drift detected in:")
        for col, psi in drifted_features:
            print(f"- {col}: PSI={psi:.4f}")
    else:
        print("No significant feature drift detected.")

    if label_drift_detected:
        print(" Label drift detected (fraud rate shift exceeded threshold).")

    if drifted_features or label_drift_detected:
        print("\n Drift threshold exceeded!")
        trigger_retraining()
    else:
        print("\n No retraining needed.")


if __name__ == "__main__":
    run_drift_check()