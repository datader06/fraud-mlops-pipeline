\# 🚀 Fraud Detection MLOps Pipeline



An end-to-end production-style machine learning pipeline for fraud detection using \*\*XGBoost, MLflow, and FastAPI\*\*, with a focus on \*\*class imbalance, threshold optimization, and deployment readiness\*\*.



\---



\## 📌 Project Overview



This project simulates a real-world fraud detection system where:



\- Data is highly imbalanced (\~3.5% fraud)

\- Predictions must be made in real-time

\- Model performance must be tracked and monitored

\- Retraining should be triggered when data drifts



\---



\## 🧠 Key Features



\- ✅ End-to-end ML pipeline (ingestion → features → training → serving)

\- ✅ Handles class imbalance using `scale\_pos\_weight`

\- ✅ Uses \*\*PR-AUC and F1\*\* instead of misleading accuracy

\- ✅ Threshold tuning for optimal fraud detection

\- ✅ MLflow experiment tracking

\- ✅ FastAPI model serving

\- 🔜 Drift detection (Evidently)

\- 🔜 Automated retraining pipeline



\---



\## 🏗️ Project Structure

fraud-mlops-pipeline/

├── src/

│ ├── data/ingest.py

│ ├── features/engineer.py

│ ├── models/train.py

│ ├── serving/app.py

│ └── monitoring/drift.py

├── configs/config.yaml

├── notebooks/

├── tests/

├── .github/workflows/

├── requirements.txt

└── README.md





\---



\## ⚙️ Tech Stack



\- Python

\- Pandas, NumPy

\- XGBoost

\- Scikit-learn

\- MLflow

\- FastAPI

\- Evidently (planned)



\---



\## 📊 Model Details



\- Model: XGBoost Classifier

\- Evaluation Metric: \*\*PR-AUC\*\*

\- Additional Metric: \*\*F1 Score (threshold optimized)\*\*

\- Imbalance Handling: `scale\_pos\_weight`



\---



\## 🚀 How to Run



\### 1. Clone repo



```bash

git clone https://github.com/datader06/fraud-mlops-pipeline.git

cd fraud-mlops-pipeline

