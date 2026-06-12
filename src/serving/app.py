import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "xgb_fraud_model.pkl"
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features.parquet"
THRESHOLD_PATH = ROOT_DIR / "models" / "optimal_threshold.txt"

# Global state
state: Dict[str, Any] = {
    "model": None,
    "feature_columns": None,
    "threshold": 0.5,
    "ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and feature columns on startup."""
    logger.info("Loading model and features...")

    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}. Run `python src/models/train.py` first.")
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        logger.error(f"Features parquet not found at {FEATURES_PATH}.")
        raise RuntimeError(f"Features file not found: {FEATURES_PATH}")

    state["model"] = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")

    df = pd.read_parquet(FEATURES_PATH)
    state["feature_columns"] = [col for col in df.columns if col != "isFraud"]
    logger.info(f"Feature columns loaded: {len(state['feature_columns'])} features.")

    if THRESHOLD_PATH.exists():
        state["threshold"] = float(THRESHOLD_PATH.read_text().strip())
        logger.info(f"Optimal threshold loaded: {state['threshold']}")
    else:
        logger.warning("optimal_threshold.txt not found — defaulting to 0.5")

    state["ready"] = True
    logger.info("API ready.")

    yield

    # Cleanup (shutdown)
    state["model"] = None
    state["ready"] = False


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using XGBoost. PR-AUC optimised model with threshold tuning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def home():
    return {
        "message": "Fraud Detection API is running",
        "status": "ready" if state["ready"] else "initialising",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Info"])
def health():
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return {"status": "healthy", "model_loaded": True, "threshold": state["threshold"]}


@app.get("/threshold", tags=["Info"])
def get_threshold():
    """Return the decision threshold currently in use."""
    return {
        "threshold": state["threshold"],
        "description": "Probability threshold above which a transaction is classified as fraud.",
    }


@app.post("/predict", tags=["Prediction"])
def predict(data: Optional[Dict[str, Any]] = Body(default={})):
    """
    Predict fraud probability for a single transaction.

    Pass a JSON object with any subset of transaction features.
    Missing features will be filled with 0 (numeric) or '0' (categorical).

    Returns:
        fraud_probability: float [0, 1]
        prediction: 0 (legit) or 1 (fraud)
        threshold_used: decision threshold applied
    """
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready. Please try again shortly.")

    try:
        data = data or {}
        input_df = pd.DataFrame([data])

        # Build full feature row in one shot (avoids DataFrame fragmentation)
        row = {col: input_df[col].iloc[0] if col in input_df.columns else 0
               for col in state["feature_columns"]}
        input_df = pd.DataFrame([row])[state["feature_columns"]]

        # Convert object columns to category (XGBoost requirement)
        for col in input_df.select_dtypes(include=["object"]).columns:
            input_df[col] = input_df[col].astype("category")

        fraud_probability = float(state["model"].predict_proba(input_df)[0][1])
        prediction = int(fraud_probability >= state["threshold"])

        return {
            "fraud_probability": round(fraud_probability, 6),
            "prediction": prediction,
            "threshold_used": state["threshold"],
            "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")
