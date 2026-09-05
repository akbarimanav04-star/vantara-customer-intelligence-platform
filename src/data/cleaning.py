"""
Load and clean the Online Retail II dataset.
"""
import pandas as pd
import numpy as np


def load_raw(path="data/raw/online_retail_II.xlsx"):
    xls = pd.ExcelFile(path)
    frames = []
    for sheet in xls.sheet_names:
        d = pd.read_excel(xls, sheet_name=sheet)
        d["SourceSheet"] = sheet
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Customer ID": "CustomerID"})

    # basic dtypes
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)

    # flag cancellations (Invoice starting with 'C')
    df["IsCancelled"] = df["Invoice"].str.startswith("C")

    # non-product stock codes (postage, fees, manual, etc.)
    non_product_codes = {
        "POST", "D", "DOT", "M", "S", "AMAZONFEE", "BANK CHARGES",
        "CRUK", "PADS", "C2", "ADJUST", "ADJUST2", "TEST001", "TEST002",
    }
    df["IsNonProduct"] = df["StockCode"].str.upper().isin(non_product_codes)

    # drop exact duplicate line items
    before = len(df)
    df = df.drop_duplicates()
    dup_dropped = before - len(df)

    # customer-level table requires a CustomerID
    df["HasCustomerID"] = df["CustomerID"].notna()

    # revenue per line
    df["LineRevenue"] = df["Quantity"] * df["Price"]

    # standardize description per stock code (mode)
    desc_lookup = (
        df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    )
    df["Description"] = df["StockCode"].map(desc_lookup).fillna(df["Description"])

    meta = {"duplicates_dropped": dup_dropped, "rows_total": len(df)}
    return df, meta


def validate(df: pd.DataFrame):
    """Basic automated data-quality checks. Raises AssertionError on violation."""
    assert df["InvoiceDate"].min().year >= 2009, "Unexpected date range"
    assert df["InvoiceDate"].max().year <= 2011, "Unexpected date range"
    null_rate_customer = df["CustomerID"].isna().mean()
    assert null_rate_customer < 0.40, f"CustomerID null rate too high: {null_rate_customer:.2%}"
    return {"customer_id_null_rate": round(null_rate_customer, 4)}


if __name__ == "__main__":
    raw = load_raw()
    cleaned, meta = clean_transactions(raw)
    checks = validate(cleaned)
    cleaned.to_parquet("data/interim/transactions_clean.parquet", index=False)
    print("Meta:", meta)
    print("Validation:", checks)
    print("Shape:", cleaned.shape)
