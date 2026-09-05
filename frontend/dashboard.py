"""
Vantara Customer Intelligence Platform - Dashboard
Run with: streamlit run frontend/dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Vantara Customer Intelligence Platform", layout="wide")

@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/customer_segments.parquet")

@st.cache_resource
def load_models():
    return (
        joblib.load("models_artifacts/preprocessor.pkl"),
        joblib.load("models_artifacts/churn_model.pkl"),
    )

df = load_data()
churn_pre, churn_model = load_models()

st.title("Vantara Customer Intelligence Platform")
st.caption("AI-powered churn prediction, lifetime value estimation, and purchase behavior analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Segmentation", "Churn Risk Leaderboard", "Revenue Trends", "Customer Lookup"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        countries = ["All"] + sorted(df["Country"].dropna().unique().tolist())
        country = st.selectbox("Filter by country", countries)
        segments = ["All"] + sorted(df["Segment"].unique().tolist())
        segment = st.selectbox("Filter by segment", segments)

    filtered = df.copy()
    if country != "All":
        filtered = filtered[filtered["Country"] == country]
    if segment != "All":
        filtered = filtered[filtered["Segment"] == segment]

    with col2:
        seg_counts = filtered["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.bar(seg_counts, x="Count", y="Segment", orientation="h", color="Segment")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment profile (averages)")
    st.dataframe(
        filtered.groupby("Segment")[["Recency", "Frequency", "Monetary", "HistoricalCLV"]].mean().round(1)
    )

    fig2 = px.scatter(filtered, x="Recency", y="Monetary", color="Segment", log_y=True,
                       hover_data=["CustomerID", "Frequency"])
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("High-value customers at highest churn risk")
    X = churn_pre.transform(df[["Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays",
                                 "AvgBasketSize", "PurchaseGapStd", "EngagementScore", "ReturnRate", "Country"]])
    df_display = df.copy()
    df_display["ChurnProbability"] = churn_model.predict_proba(X)[:, 1]
    leaderboard = df_display.sort_values(["ChurnProbability", "HistoricalCLV"], ascending=[False, False])
    st.dataframe(
        leaderboard[["CustomerID", "Country", "Segment", "ChurnProbability", "HistoricalCLV", "Recency", "Frequency"]]
        .head(50).style.format({"ChurnProbability": "{:.1%}", "HistoricalCLV": "${:,.0f}"})
    )
    csv = leaderboard.to_csv(index=False).encode("utf-8")
    st.download_button("Download full leaderboard as CSV", csv, "churn_leaderboard.csv")

with tab3:
    st.subheader("Historical CLV distribution by segment")
    fig3 = px.box(df, x="Segment", y="HistoricalCLV", color="Segment", log_y=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.subheader("Revenue concentration")
    seg_rev = df.groupby("Segment")["HistoricalCLV"].sum().reset_index()
    fig4 = px.pie(seg_rev, names="Segment", values="HistoricalCLV", title="Share of total historical revenue by segment")
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.subheader("Look up an individual customer")
    cid = st.selectbox("Customer ID", df["CustomerID"].sort_values().tolist())
    row = df[df["CustomerID"] == cid].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment", row["Segment"])
    c2.metric("Recency (days)", int(row["Recency"]))
    c3.metric("Frequency (orders)", int(row["Frequency"]))
    c4.metric("Historical CLV", f"${row['HistoricalCLV']:,.0f}")
    Xrow = churn_pre.transform(pd.DataFrame([row])[["Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays",
                                                     "AvgBasketSize", "PurchaseGapStd", "EngagementScore", "ReturnRate", "Country"]])
    proba = churn_model.predict_proba(Xrow)[0][1]
    st.metric("Churn probability", f"{proba:.1%}")

    st.markdown("**Upload a CSV to batch-score customers via the API**")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        st.info("Send this file to POST /predict/batch on the running API to get churn probabilities back.")
