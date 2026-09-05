import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SEQ_LEN = 15
MIN_ORDERS = 3


def build_sequences(df):
    """Per-customer, time-ordered sequence of (order value, gap-in-days) pairs."""
    df = df[df["HasCustomerID"] & ~df["IsCancelled"] & ~df["IsNonProduct"]]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    order_level = df.groupby(["CustomerID", "Invoice"]).agg(
        OrderValue=("LineRevenue", "sum"),
        OrderDate=("InvoiceDate", "min"),
    ).reset_index()

    sequences, labels, cust_ids = [], [], []
    max_date = order_level["OrderDate"].max()

    for cid, g in order_level.groupby("CustomerID"):
        g = g.sort_values("OrderDate")
        if len(g) < MIN_ORDERS:
            continue
        # use all orders except the last as the sequence; label = did they churn (90d rule, same as main model)
        dates = g["OrderDate"].values
        values = np.log1p(g["OrderValue"].values)
        gaps = np.diff(dates).astype("timedelta64[D]").astype(float)
        gaps = np.insert(gaps, 0, 0)

        seq = np.stack([values, gaps], axis=1)[-(MAX_SEQ_LEN + 1):-1]  # last N orders, excluding the very last
        if len(seq) < 2:
            continue
        last_order_date = pd.Timestamp(g["OrderDate"].iloc[-1])
        churned = int((max_date - last_order_date).days > 90)

        sequences.append(seq)
        labels.append(churned)
        cust_ids.append(cid)

    return sequences, np.array(labels), cust_ids


def main():
    df = pd.read_parquet("data/interim/transactions_clean.parquet")
    sequences, labels, cust_ids = build_sequences(df)
    print(f"Built {len(sequences)} customer sequences. Churn rate: {labels.mean():.2%}")

    # scale using flattened training stats (fit only after split to avoid leakage)
    idx = np.arange(len(sequences))
    idx_train, idx_test, y_train, y_test = train_test_split(idx, labels, test_size=0.2, stratify=labels, random_state=42)

    flat_train = np.concatenate([sequences[i] for i in idx_train], axis=0)
    scaler = StandardScaler().fit(flat_train)

    def scale_and_pad(indices):
        scaled = [scaler.transform(sequences[i]) for i in indices]
        return pad_sequences(scaled, maxlen=MAX_SEQ_LEN, dtype="float32", padding="pre", value=0.0)

    X_train = scale_and_pad(idx_train)
    X_test = scale_and_pad(idx_test)

    model = models.Sequential([
        layers.Input(shape=(MAX_SEQ_LEN, 2)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.15, epochs=60, batch_size=32, callbacks=[es], verbose=0)

    proba = model.predict(X_test, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    results = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "n_sequences": len(sequences),
        "n_test": len(idx_test),
    }
    print("LSTM results:", results)

    model.save("models_artifacts/lstm_churn.keras")
    with open("reports/lstm_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
