import subprocess
import pandas as pd
import numpy as np
import yaml
import argparse
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


def load_current_data(config, current_data_path=None):
    """
    Load current (new) data for drift comparison.

    Priority:
      1. --current-data <path>  passed via CLI  → use that file directly
      2. data/processed/current_batch.parquet    → drop new data here
      3. Fallback: last 30% of features.parquet  → simulated batch
    """
    # Priority 1 — explicit path from CLI
    if current_data_path:
        path = Path(current_data_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Provided current data path not found: {path}")
        print(f"Using provided current data: {path}")
        return pd.read_parquet(path)

    # Priority 2 — drop-in file
    drop_in = ROOT_DIR / "data" / "processed" / "current_batch.parquet"
    if drop_in.exists():
        print(f"Using drop-in current batch: {drop_in}")
        return pd.read_parquet(drop_in)

    # Priority 3 — simulate from training data
    print("No current batch found — simulating with last 30% of features.parquet")
    df = pd.read_parquet(ROOT_DIR / config["data"]["processed_path"])
    split_idx = int(len(df) * 0.7)
    return df.iloc[split_idx:].copy()


def run_drift_check(current_data_path=None):
    config = load_config()

    reference_path = ROOT_DIR / config["data"]["reference_path"]

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {reference_path}.\n"
            "Run: python src/data/save_reference.py"
        )

    print(f"Loading reference data from: {reference_path}")
    reference_df = pd.read_parquet(reference_path)

    current_df = load_current_data(config, current_data_path)

    config_drift = config["drift"]
    min_batch = config_drift.get("min_batch_size", 500)
    if len(current_df) < min_batch:
        print(f"\nWarning: current batch has only {len(current_df)} rows "
              f"(minimum recommended: {min_batch}). Results may be unreliable.")

    print(f"\nReference shape : {reference_df.shape}")
    print(f"Current shape   : {current_df.shape}")

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

    # ------------------------------------------------------------------
    # Evidently HTML report
    # ------------------------------------------------------------------
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_features, current_data=current_features)

    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    output_path = reports_dir / "drift_report.html"
    report.save_html(str(output_path))

    print(f"\n Drift report saved at: {output_path}")

    # ------------------------------------------------------------------
    # PSI check on numeric columns
    # ------------------------------------------------------------------
    psi_threshold = config_drift["psi_threshold"]
    top_n = config_drift.get("top_n_features", 20)

    numeric_cols = reference_features.select_dtypes(include=["number"]).columns.tolist()
    drifted_features = []

    print("\n PSI results:")
    for col in numeric_cols[:top_n]:
        psi = calculate_psi(reference_features[col], current_features[col])
        if pd.notna(psi):
            status = "🔴 DRIFT" if psi > psi_threshold else ("⚠️  MILD" if psi > 0.1 else "✅")
            print(f"  {col:<25} PSI={psi:.4f}  {status}")
            if psi > psi_threshold:
                drifted_features.append((col, psi))

    # ------------------------------------------------------------------
    # Label drift check
    # ------------------------------------------------------------------
    label_drift_detected = False
    if reference_target is not None and current_target is not None:
        ref_rate = reference_target.mean()
        cur_rate = current_target.mean()
        label_shift = abs(cur_rate - ref_rate)

        print("\n Label drift check:")
        print(f"  Reference fraud rate : {ref_rate:.4f} ({ref_rate*100:.2f}%)")
        print(f"  Current fraud rate   : {cur_rate:.4f} ({cur_rate*100:.2f}%)")
        print(f"  Absolute shift       : {label_shift:.4f}")

        if label_shift > 0.02:
            label_drift_detected = True
            print("  🔴 Label drift detected!")
        else:
            print("  ✅ No label drift")

    # ------------------------------------------------------------------
    # Summary and retraining trigger
    # ------------------------------------------------------------------
    print("\n" + "="*50)
    print(" DRIFT SUMMARY")
    print("="*50)

    if drifted_features:
        print("🔴 Significant feature drift in:")
        for col, psi in drifted_features:
            print(f"   - {col}: PSI={psi:.4f}")
    else:
        print("✅ No significant feature drift detected.")

    if label_drift_detected:
        print("🔴 Label drift: fraud rate shifted beyond threshold.")
    elif reference_target is not None:
        print("✅ No label drift detected.")

    if drifted_features or label_drift_detected:
        print("\n⚠️  Drift threshold exceeded → triggering retraining...")
        trigger_retraining()
    else:
        print("\n✅ Model is stable. No retraining needed.")

    print(f"\n📄 Open the full report: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run drift detection against the reference dataset."
    )
    parser.add_argument(
        "--current-data",
        type=str,
        default=None,
        help=(
            "Path to the current (new) data parquet file to check for drift. "
            "Can be absolute or relative to the project root. "
            "If not provided, falls back to data/processed/current_batch.parquet "
            "or simulates using the last 30%% of features.parquet."
        ),
    )
    args = parser.parse_args()
    run_drift_check(current_data_path=args.current_data)