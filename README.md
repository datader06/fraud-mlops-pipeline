# 🚀 Fraud Detection MLOps Pipeline

An end-to-end **Machine Learning + MLOps pipeline** for fraud detection using XGBoost, featuring automated training, API deployment, drift monitoring, and retraining triggers.

---

## 📌 Project Overview

This project builds a **production-style fraud detection system** that:

* Processes raw transaction data
* Engineers meaningful features
* Trains a machine learning model
* Serves predictions via API
* Monitors data drift
* Automatically triggers retraining when drift is detected

---

## 🧠 Key Features

✅ End-to-End ML Pipeline
✅ XGBoost Model with Imbalance Handling
✅ Threshold Optimization (F1-based)
✅ MLflow Experiment Tracking
✅ FastAPI Model Serving
✅ Drift Detection using Evidently + PSI
✅ Automated Retraining Trigger
✅ Modular & Scalable Architecture

---

## 🏗️ Project Architecture

```text
Data → Feature Engineering → Model Training → MLflow Tracking
     → FastAPI API → Drift Detection → Retraining Trigger
```

---

## 📂 Project Structure

```text
fraud-mlops-pipeline/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
│
├── models/
│   ├── xgb_fraud_model.pkl
│   ├── category_maps.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── serving/
│   └── monitoring/
│
├── reports/
│   └── drift_report.html
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/datader06/fraud-mlops-pipeline.git
cd fraud-mlops-pipeline
```

---

### 2. Create Virtual Environment (Python 3.11 recommended)

```bash
py -3.11 -m venv venv311
.\venv311\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training

```bash
python src/models/train.py
```

✔ Outputs:

* Trained model (`models/`)
* Encoders & feature columns
* MLflow logs

---

## 📊 MLflow Tracking

```bash
python -m mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## 🌐 API Serving (FastAPI)

```bash
uvicorn src.serving.app:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

### Example Request

```json
{
  "TransactionDT": 86400,
  "TransactionAmt": 150.5,
  "ProductCD": "W",
  "card1": 13926,
  "card2": 321,
  "card4": "visa",
  "card6": "credit",
  "addr1": 315,
  "P_emaildomain": "gmail.com"
}
```

---

## 📈 Drift Detection

```bash
python src/data/save_reference.py
python src/monitoring/drift.py
```

✔ Generates:

* Drift report (`reports/drift_report.html`)
* PSI-based drift metrics

---

## 🔄 Automated Retraining

If drift exceeds threshold:

```text
Drift Detected → Trigger Retraining → Save New Model → Log to MLflow
```

---

## 📊 Evaluation Metrics

* PR-AUC (Primary metric)
* F1 Score (Threshold optimized)
* Precision-Recall Curve

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* FastAPI
* MLflow
* Evidently AI
* Uvicorn

---

## 🎯 Future Improvements

* Docker containerization
* CI/CD pipeline automation
* Real-time streaming data support
* Model versioning & rollback
* Alerting system for drift

---

## 👨‍💻 Author

**Neil Mhatre**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
