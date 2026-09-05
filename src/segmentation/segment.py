import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

FEATURES = ["Recency", "Frequency", "Monetary"]


def label_segment(row):
    if row["Recency"] <= row["Recency_med"] and row["Frequency"] >= row["Frequency_med"] and row["Monetary"] >= row["Monetary_med"]:
        return "Champions"
    if row["Recency"] > row["Recency_med"] and row["Monetary"] >= row["Monetary_med"]:
        return "Lapsed High-Value"
    if row["Recency"] <= row["Recency_med"] and row["Frequency"] < row["Frequency_med"]:
        return "New / Occasional"
    return "At-Risk Low-Value"


def main():
    df = pd.read_parquet("data/processed/customer_features.parquet")
    X = df[FEATURES].copy()
    # log-transform monetary (heavy skew) before scaling
    X["Monetary"] = np.log1p(X["Monetary"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, best_score, best_model = None, -1, None
    scores = {}
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        scores[k] = {"silhouette": round(sil, 4), "davies_bouldin": round(db, 4)}
        if sil > best_score:
            best_k, best_score, best_model = k, sil, km

    print("Cluster quality by k:", scores)
    print("Chosen k:", best_k, "silhouette:", round(best_score, 4))

    df["Cluster"] = best_model.predict(X_scaled)

    meds = df[["Recency", "Frequency", "Monetary"]].median().add_suffix("_med")
    df_labeled = df.join(pd.DataFrame([meds.to_dict()] * len(df), index=df.index))
    df["Segment"] = df_labeled.apply(label_segment, axis=1)

    profile = df.groupby("Segment")[["Recency", "Frequency", "Monetary", "HistoricalCLV"]].mean().round(1)
    print("\nSegment profile:\n", profile)

    joblib.dump(scaler, "models_artifacts/segment_scaler.pkl")
    joblib.dump(best_model, "models_artifacts/segment_kmeans.pkl")
    df.to_parquet("data/processed/customer_segments.parquet", index=False)
    profile.to_csv("reports/segment_profile.csv")


if __name__ == "__main__":
    main()
