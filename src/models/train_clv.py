import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FEATURES_NUM = ["Recency", "Frequency", "AvgOrderValue", "TenureDays", "AvgBasketSize", "PurchaseGapStd", "ReturnRate"]
FEATURES_CAT = ["Country"]
TARGET = "HistoricalCLV"


def main():
    df = pd.read_parquet("data/processed/customer_features.parquet")
    top_countries = df["Country"].value_counts().nlargest(8).index
    df["Country"] = df["Country"].where(df["Country"].isin(top_countries), "Other")

    # log-transform the heavily skewed monetary target for stabler training
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = np.log1p(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    results = {}
    models = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoostRegressor": XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_train_t, y_train)
        pred_log = m.predict(X_test_t)
        pred = np.expm1(pred_log)
        actual = np.expm1(y_test)
        results[name] = {
            "MAE": round(mean_absolute_error(actual, pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(actual, pred)), 2),
            "R2": round(r2_score(actual, pred), 4),
        }
        trained[name] = m
        print(name, results[name])

    best_name = max(results, key=lambda k: results[k]["R2"])
    print("Best CLV model:", best_name)

    joblib.dump(pre, "models_artifacts/clv_preprocessor.pkl")
    joblib.dump(trained[best_name], "models_artifacts/clv_model.pkl")
    with open("reports/clv_model_comparison.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)


if __name__ == "__main__":
    main()
