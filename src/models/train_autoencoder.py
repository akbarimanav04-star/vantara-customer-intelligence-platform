import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

SPEND_FEATURES = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "AvgBasketSize", "PurchaseGapStd"]


def main():
    df = pd.read_parquet("data/processed/customer_features.parquet")
    X = df[SPEND_FEATURES].copy()
    X["Monetary"] = np.log1p(X["Monetary"])
    X["AvgOrderValue"] = np.log1p(X["AvgOrderValue"])
    X["AvgBasketSize"] = np.log1p(X["AvgBasketSize"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype("float32")

    input_dim = X_scaled.shape[1]
    autoencoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(4, activation="relu", name="bottleneck"),
        layers.Dense(16, activation="relu"),
        layers.Dense(input_dim, activation="linear"),
    ])
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_scaled, X_scaled, epochs=60, batch_size=32, validation_split=0.15, verbose=0)

    recon = autoencoder.predict(X_scaled, verbose=0)
    recon_error = np.mean((X_scaled - recon) ** 2, axis=1)
    threshold = np.percentile(recon_error, 95)
    df["ReconstructionError"] = recon_error
    df["AnomalyFlag"] = (recon_error > threshold).astype(int)

    print(f"Reconstruction error threshold (95th pct): {threshold:.4f}")
    print(f"Flagged {df['AnomalyFlag'].sum()} anomalous customers out of {len(df)}")
    print(df[df["AnomalyFlag"] == 1][["CustomerID", "Recency", "Frequency", "Monetary", "ReconstructionError"]].sort_values("ReconstructionError", ascending=False).head(10))

    autoencoder.save("models_artifacts/autoencoder.keras")
    joblib.dump(scaler, "models_artifacts/autoencoder_scaler.pkl")
    df[["CustomerID", "ReconstructionError", "AnomalyFlag"]].to_parquet("data/processed/anomaly_flags.parquet", index=False)
    with open("reports/autoencoder_summary.json", "w") as f:
        json.dump({"threshold_95pct": float(threshold), "n_flagged": int(df["AnomalyFlag"].sum()), "n_total": len(df)}, f, indent=2)


if __name__ == "__main__":
    main()
