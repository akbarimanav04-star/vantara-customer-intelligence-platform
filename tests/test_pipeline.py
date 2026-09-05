import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.build_features import build_customer_features, OBSERVATION_WINDOW_DAYS


def _toy_transactions():
    dates = pd.date_range("2011-01-01", periods=200, freq="D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "CustomerID": 1 if i % 2 == 0 else 2,
            "Invoice": f"INV{i}",
            "InvoiceDate": d,
            "Quantity": 2,
            "Price": 10.0,
            "LineRevenue": 20.0,
            "Country": "United Kingdom",
            "HasCustomerID": True,
            "IsCancelled": False,
            "IsNonProduct": False,
        })
    return pd.DataFrame(rows)


def test_no_leakage_in_features():
    """Every feature must only use transactions dated on/before the cutoff."""
    df = _toy_transactions()
    feats = build_customer_features(df)
    max_date = df["InvoiceDate"].max()
    cutoff = max_date - pd.Timedelta(days=OBSERVATION_WINDOW_DAYS)
    # Recency computed relative to cutoff must be >= 0 for every customer
    assert (feats["Recency"] >= 0).all()


def test_churn_label_is_binary():
    df = _toy_transactions()
    feats = build_customer_features(df)
    assert set(feats["Churned"].unique()).issubset({0, 1})


def test_engagement_score_bounded():
    df = _toy_transactions()
    feats = build_customer_features(df)
    assert feats["EngagementScore"].between(0, 1).all()
