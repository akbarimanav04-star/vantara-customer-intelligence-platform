import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

FEATURES_NUM = [
    "Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays",
    "AvgBasketSize", "PurchaseGapStd", "EngagementScore", "ReturnRate",
]
FEATURES_CAT = ["Country"]
TARGET = "Churned"


def main():
    df = pd.read_parquet("data/processed/customer_features.parquet")
    top_countries = df["Country"].value_counts().nlargest(8).index
    df["Country"] = df["Country"].where(df["Country"].isin(top_countries), "Other")

    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # reuse the same fitted preprocessor as the classical churn model for a fair, apples-to-apples comparison
    pre = joblib.load("models_artifacts/preprocessor.pkl")
    X_train_t = pre.transform(X_train).astype("float32")
    X_val_t = pre.transform(X_val).astype("float32")
    X_test_t = pre.transform(X_test).astype("float32")

    n_features = X_train_t.shape[1]
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])

    es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    history = model.fit(
        X_train_t, y_train.values, validation_data=(X_val_t, y_val.values),
        epochs=100, batch_size=32, callbacks=[es], verbose=0,
    )
    print(f"Trained for {len(history.history['loss'])} epochs (early stopping)")

    proba = model.predict(X_test_t, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    results = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    print("ANN results:", results)

    model.save("models_artifacts/ann_churn.keras")
    with open("reports/ann_churn_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
