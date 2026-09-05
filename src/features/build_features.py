"""
Build the customer-level feature table: RFM, CLV, behavioral & churn-label features.

Churn label definition (business rule, documented per PRD Section 7):
A customer is labeled CHURNED if they made no purchase in the final
OBSERVATION_WINDOW_DAYS of the dataset's timeline, having been active before that
cutoff. This mirrors a "no purchase in the last N days" operational churn
definition retailers use in practice.
"""
import pandas as pd
import numpy as np

OBSERVATION_WINDOW_DAYS = 90  # last 90 days of the dataset = the "did they come back" window


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["HasCustomerID"] & ~df["IsCancelled"] & ~df["IsNonProduct"]].copy()
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    max_date = df["InvoiceDate"].max()
    cutoff = max_date - pd.Timedelta(days=OBSERVATION_WINDOW_DAYS)

    # Split into "history" (before cutoff, used to build features) and
    # "future" (cutoff to end, used only to derive the churn label) -> avoids leakage.
    hist = df[df["InvoiceDate"] <= cutoff]
    future = df[df["InvoiceDate"] > cutoff]

    snapshot_date = cutoff

    # ---- RFM ----
    grp = hist.groupby("CustomerID")
    rfm = grp.agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineRevenue", "sum"),
        AvgOrderValue=("LineRevenue", "mean"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
    ).reset_index()

    rfm["TenureDays"] = (snapshot_date - rfm["FirstPurchase"]).dt.days
    rfm["AvgBasketSize"] = hist.groupby(["CustomerID", "Invoice"])["Quantity"].sum().groupby("CustomerID").mean().values

    # purchase gap variability
    def gap_std(g):
        dates = g["InvoiceDate"].drop_duplicates().sort_values()
        if len(dates) < 3:
            return 0.0
        return dates.diff().dt.days.dropna().std()

    gap = hist.groupby("CustomerID").apply(gap_std, include_groups=False).rename("PurchaseGapStd").reset_index()
    rfm = rfm.merge(gap, on="CustomerID", how="left")
    rfm["PurchaseGapStd"] = rfm["PurchaseGapStd"].fillna(0)

    # return rate (uses full history incl. cancellations, mapped back to customer)
    all_hist_incl_cancel = df_full_for_returns = None  # placeholder, computed below in caller

    # country (mode)
    country = hist.groupby("CustomerID")["Country"].agg(lambda s: s.mode().iloc[0]).reset_index()
    rfm = rfm.merge(country, on="CustomerID", how="left")

    # ---- Historical CLV (regression target): total monetary value in the FULL dataset ----
    full_value = df.groupby("CustomerID")["LineRevenue"].sum().rename("HistoricalCLV").reset_index()
    rfm = rfm.merge(full_value, on="CustomerID", how="left")

    # ---- Churn label: no purchase in the future window despite being active in history ----
    active_in_future = set(future["CustomerID"].unique())
    rfm["Churned"] = (~rfm["CustomerID"].isin(active_in_future)).astype(int)

    # engagement score (0-1 composite)
    def norm(s):
        rng = (s.max() - s.min())
        return (s - s.min()) / rng if rng > 0 else s * 0
    rfm["EngagementScore"] = (
        (1 - norm(rfm["Recency"])) * 0.4 + norm(rfm["Frequency"]) * 0.3 + norm(rfm["Monetary"]) * 0.3
    )

    rfm = rfm.drop(columns=["FirstPurchase", "LastPurchase"])
    return rfm


def add_return_rate(df_raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Return rate uses cancelled invoices from the full (non-future-filtered) history."""
    base = df_raw[df_raw["HasCustomerID"] & ~df_raw["IsNonProduct"]]
    total_lines = base.groupby("CustomerID").size().rename("TotalLines")
    returned_lines = base[base["IsCancelled"]].groupby("CustomerID").size().rename("ReturnedLines")
    rr = pd.concat([total_lines, returned_lines], axis=1).fillna(0)
    rr["ReturnRate"] = rr["ReturnedLines"] / rr["TotalLines"].replace(0, np.nan)
    rr["ReturnRate"] = rr["ReturnRate"].fillna(0)
    features = features.merge(rr[["ReturnRate"]], left_on="CustomerID", right_index=True, how="left")
    features["ReturnRate"] = features["ReturnRate"].fillna(0)
    return features


if __name__ == "__main__":
    raw = pd.read_parquet("data/interim/transactions_clean.parquet")
    feats = build_customer_features(raw)
    feats = add_return_rate(raw, feats)
    feats.to_parquet("data/processed/customer_features.parquet", index=False)
    print("Feature table shape:", feats.shape)
    print(feats["Churned"].value_counts(normalize=True))
    print(feats.describe().T)
