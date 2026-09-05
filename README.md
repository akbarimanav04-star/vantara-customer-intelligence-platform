# Vantara Customer Intelligence Platform

AI-powered analytics that predicts customer churn, estimates customer lifetime value (CLV),
and segments customers by purchase behavior — trained on the UCI **Online Retail II** dataset.

## What's already built and trained in this repo

- `src/data/cleaning.py` — loads both sheets of the raw Excel file, deduplicates, flags
  cancellations/non-product codes, validates the result.
- `src/features/build_features.py` — builds RFM, tenure, basket, engagement, return-rate
  features and a leakage-safe churn label (90-day forward-looking window).
- `src/models/train_churn.py` — trains Logistic Regression, Decision Tree, Random Forest,
  XGBoost, LightGBM with SMOTE; picks the best by ROC-AUC. **Best: Random Forest, ROC-AUC 0.79.**
- `src/models/train_clv.py` — trains Random Forest and XGBoost regressors on log-CLV.
  **Best: XGBoost, R² 0.92.**
- `src/models/train_ann.py` — feed-forward ANN (Keras) on the same feature table.
  **ROC-AUC 0.808 — the single best-performing churn model overall.**
- `src/models/train_lstm.py` — LSTM over each customer's order-value/gap sequence.
  ROC-AUC 0.778 on 3,289 customers with 3+ orders.
- `src/models/train_autoencoder.py` — unsupervised anomaly detector on spending patterns;
  flags the top 5% by reconstruction error (263 customers) for manual review.
- `src/segmentation/segment.py` — K-Means segmentation (k chosen by silhouette score) into
  4 business-labeled segments: Champions, Lapsed High-Value, New/Occasional, At-Risk Low-Value.
- `src/explainability/explain.py` — SHAP global + per-customer explanations.
- `src/explainability/explain_lime_pdp.py` — LIME cross-check + Partial Dependence Plots.
- `api/main.py` — FastAPI service exposing `/predict/churn`, `/predict/clv`,
  `/predict/segment`, `/predict/batch`.
- `frontend/dashboard.py` — Streamlit dashboard (segmentation, churn leaderboard, revenue
  trends, customer lookup).
- `tests/test_pipeline.py` — leakage and sanity tests (`pytest tests/`).

All trained artifacts are already saved under `models_artifacts/` — you do not need to
retrain anything to run the API or dashboard.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt

# API
uvicorn api.main:app --reload   # http://127.0.0.1:8000/docs

# Dashboard (in a second terminal)
streamlit run frontend/dashboard.py   # http://localhost:8501
```

## Re-running the pipeline from scratch

```bash
# 1. put online_retail_II.xlsx in data/raw/
python src/data/cleaning.py
python src/features/build_features.py
python src/models/train_churn.py
python src/models/train_clv.py
python src/segmentation/segment.py
python src/models/train_ann.py
python src/models/train_lstm.py
python src/models/train_autoencoder.py
python src/explainability/explain.py
python src/explainability/explain_lime_pdp.py
python src/data/eda_charts.py
```

## Docker

```bash
docker-compose up --build
```

## Results summary (on the real dataset)

| Model | Task | Key metric |
|---|---|---|
| ANN (Keras) | Churn classification | **ROC-AUC 0.808** (best overall) |
| Random Forest | Churn classification (production default — no TF dependency) | ROC-AUC 0.793, Recall 0.74 |
| LSTM | Sequential churn (order-sequence based) | ROC-AUC 0.778 |
| Autoencoder | Anomaly detection | 263/5,256 customers flagged (95th pct. threshold) |
| XGBoost | CLV regression | R² 0.92, MAE $929 |
| K-Means (k=3) | Segmentation | Silhouette 0.41 |

The API defaults to Random Forest for the churn endpoint to keep the deployment lightweight
(no TensorFlow runtime dependency). Swap in `models_artifacts/ann_churn.keras` for a ~1.5-point
ROC-AUC gain if you don't mind the heavier container — load it with
`tf.keras.models.load_model(...)` and predict with `.predict()` instead of `.predict_proba()`.

See `reports/` for full metric JSON files, SHAP/LIME/PDP charts, and segment profiles.

## Dataset credit

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. CC BY 4.0.
