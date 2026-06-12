"""
train.py — Model Training with selectable algorithm
────────────────────────────────────────────────────
Usage:
  python src/models/train.py                        # default: xgboost
  python src/models/train.py --model xgboost
  python src/models/train.py --model random_forest
  python src/models/train.py --model lightgbm
  python src/models/train.py --model logistic
"""
import argparse
import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

import mlflow
import mlflow.sklearn
import mlflow.xgboost

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT_DIR / "configs" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximises F1 on validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def build_model(model_name: str, scale_pos_weight: float, config: dict):
    """
    Return a configured (but untrained) model.

    Supported models
    ─────────────────
    xgboost       Best for tabular fraud data. Handles categories natively.
                  → Best choice for this dataset (highly recommended)

    lightgbm      Faster than XGBoost, similar accuracy. Good for very large data.
                  → Good alternative if training is too slow

    random_forest Simpler, more interpretable. Slower at inference.
                  → Good baseline / explainability

    logistic      Linear model, very fast, limited accuracy on complex patterns.
                  → Use as a baseline benchmark only
    """
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            learning_rate=config["model"]["learning_rate"],
            eval_metric=config["model"]["eval_metric"],
            subsample=config["model"]["subsample"],
            colsample_bytree=config["model"]["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            enable_categorical=True,
            tree_method="hist",
            random_state=42,
        )

    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            learning_rate=config["model"]["learning_rate"],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
        )

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=config["model"]["max_depth"],
            class_weight="balanced",   # handles imbalance
            random_state=42,
            n_jobs=-1,
        )

    elif model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="saga",
            )),
        ])

    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            "Choose from: xgboost, lightgbm, random_forest, logistic"
        )


def log_model_to_mlflow(model, model_name, X_train, config):
    """Log model artifact to MLflow using the right logger per model type."""
    from mlflow.models.signature import infer_signature
    sig = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
    
    # Use sklearn log_model to bypass native C++ serialization bug with categorical splits
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=sig,
        registered_model_name=config["mlflow"]["registered_model_name"] if model_name == "xgboost" else f"{config['mlflow']['registered_model_name']}-{model_name}",
        input_example=X_train.iloc[:5],
    )


def train_model(model_name: str = "xgboost"):
    config = load_config()

    print(f"\n{'='*50}")
    print(f" MODEL: {model_name.upper()}")
    print(f"{'='*50}\n")

    print("Loading features...")
    df = pd.read_parquet(ROOT_DIR / config["data"]["processed_path"])
    # Downsample for quick training
    df = df.sample(n=5000, random_state=42)
    print(f"Dataset shape: {df.shape}")

    y = df["isFraud"]
    X = df.drop(columns=["isFraud"])

    # For non-XGBoost models: drop category columns (they need numeric input)
    if model_name in ("random_forest", "logistic"):
        cat_cols = X.select_dtypes(include=["category", "object"]).columns
        X = X.drop(columns=cat_cols)
        print(f"Dropped {len(cat_cols)} categorical columns for {model_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fraud_count  = (y_train == 1).sum()
    normal_count = (y_train == 0).sum()
    scale_pos_weight = normal_count / fraud_count
    print(f"Scale Pos Weight: {scale_pos_weight:.2f}  "
          f"({normal_count:,} normal / {fraud_count:,} fraud)")

    # MLflow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_name) as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_params({
            "model_type": model_name,
            "n_estimators": config["model"]["n_estimators"],
            "max_depth": config["model"]["max_depth"],
            "learning_rate": config["model"]["learning_rate"],
            "scale_pos_weight": round(scale_pos_weight, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "fraud_rate_train": round(float(y_train.mean()), 4),
        })

        model = build_model(model_name, scale_pos_weight, config)

        print("Training model...")
        model.fit(X_train, y_train)

        preds_proba = model.predict_proba(X_test)[:, 1]

        # Threshold optimisation
        best_threshold, best_f1_thresh = find_optimal_threshold(y_test, preds_proba)
        print(f"Optimal threshold: {best_threshold:.4f}  (F1={best_f1_thresh:.4f})")

        pred_labels_default = (preds_proba >= 0.5).astype(int)
        pred_labels_optimal  = (preds_proba >= best_threshold).astype(int)

        # Metrics
        roc_auc  = roc_auc_score(y_test, preds_proba)
        pr_auc   = average_precision_score(y_test, preds_proba)

        precision_default = precision_score(y_test, pred_labels_default, zero_division=0)
        recall_default    = recall_score(y_test, pred_labels_default, zero_division=0)
        f1_default        = f1_score(y_test, pred_labels_default, zero_division=0)

        precision_optimal = precision_score(y_test, pred_labels_optimal, zero_division=0)
        recall_optimal    = recall_score(y_test, pred_labels_optimal, zero_division=0)
        f1_optimal        = f1_score(y_test, pred_labels_optimal, zero_division=0)

        print(f"\n{'='*40}")
        print(f" RESULTS — {model_name.upper()}")
        print(f"{'='*40}")
        print(f"  ROC-AUC              : {roc_auc:.4f}")
        print(f"  PR-AUC  (key metric) : {pr_auc:.4f}")
        print(f"\n  @threshold=0.5")
        print(f"    Precision          : {precision_default:.4f}")
        print(f"    Recall             : {recall_default:.4f}")
        print(f"    F1 Score           : {f1_default:.4f}")
        print(f"\n  @threshold={best_threshold:.4f} (optimal F1)")
        print(f"    Precision          : {precision_optimal:.4f}")
        print(f"    Recall             : {recall_optimal:.4f}")
        print(f"    F1 Score           : {f1_optimal:.4f}")

        cm = confusion_matrix(y_test, pred_labels_optimal)
        print(f"\n  Confusion Matrix (optimal threshold):")
        print(f"    TN={cm[0,0]:>6,}  FP={cm[0,1]:>6,}")
        print(f"    FN={cm[1,0]:>6,}  TP={cm[1,1]:>6,}")

        mlflow.log_metrics({
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1_default": round(f1_default, 4),
            "f1_optimal": round(f1_optimal, 4),
            "precision_optimal": round(precision_optimal, 4),
            "recall_optimal": round(recall_optimal, 4),
            "optimal_threshold": round(best_threshold, 4),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        })

        log_model_to_mlflow(model, model_name, X_train, config)

        # Save locally for FastAPI
        model_dir = ROOT_DIR / "models"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(model, model_dir / "xgb_fraud_model.pkl")
        (model_dir / "optimal_threshold.txt").write_text(str(round(best_threshold, 6)))
        (model_dir / "model_type.txt").write_text(model_name)

        print(f"\nModel saved to models/xgb_fraud_model.pkl")
        print(f"Optimal threshold saved to models/optimal_threshold.txt")
        print(f"View in MLflow: {config['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "random_forest", "logistic"],
        help=(
            "Model to train:\n"
            "  xgboost       → Best accuracy for this dataset (recommended)\n"
            "  lightgbm      → Faster training, similar accuracy\n"
            "  random_forest → More interpretable, good baseline\n"
            "  logistic      → Fastest, weakest — use as benchmark only"
        ),
    )
    args = parser.parse_args()
    train_model(model_name=args.model)