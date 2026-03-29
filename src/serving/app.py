import joblib
import pandas as pd
from fastapi import FastAPI
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

app = FastAPI(title="Fraud Detection API")

# Load model
model = joblib.load(ROOT_DIR / "models" / "xgb_fraud_model.pkl")

# Load training feature columns from processed data
df = pd.read_parquet(ROOT_DIR / "data" / "processed" / "features.parquet")
feature_columns = [col for col in df.columns if col != "isFraud"]


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(data: dict):
    # Convert input to dataframe
    input_df = pd.DataFrame([data])

    # Add missing columns with default 0
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only training columns, same order
    input_df = input_df[feature_columns]

    # Convert object columns to category if needed
    for col in input_df.select_dtypes(include=["object"]).columns:
        input_df[col] = input_df[col].astype("category")

    fraud_probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(fraud_probability >= 0.8013)

    return {
        "fraud_probability": fraud_probability,
        "prediction": prediction
    }