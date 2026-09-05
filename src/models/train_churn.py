import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

FEATURES_NUM = [
    "Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays",
    "AvgBasketSize", "PurchaseGapStd", "EngagementScore", "ReturnRate",
]
FEATURES_CAT = ["Country"]
TARGET = "Churned"


def load_data():
    df = pd.read_parquet("data/processed/customer_features.parquet")
    top_countries = df["Country"].value_counts().nlargest(8).index
    df["Country"] = df["Country"].where(df["Country"].isin(top_countries), "Other")
    return df


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])


def evaluate(name, model, X_test, y_test, results):
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    print(name, results[name])


def main():
    df = load_data()
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_t, y_train)

    results = {}
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42,
                                  eval_metric="logloss", n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1),
    }

    trained = {}
    for name, m in models.items():
        m.fit(X_train_res, y_train_res)
        trained[name] = m
        evaluate(name, m, X_test_t, y_test, results)

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained[best_name]
    print("\nBest model by ROC-AUC:", best_name)

    cm = confusion_matrix(y_test, best_model.predict(X_test_t))
    print("Confusion matrix (best model):\n", cm)

    # persist
    joblib.dump(pre, "models_artifacts/preprocessor.pkl")
    joblib.dump(best_model, "models_artifacts/churn_model.pkl")
    joblib.dump({"features_num": FEATURES_NUM, "features_cat": FEATURES_CAT}, "models_artifacts/feature_spec.pkl")
    with open("reports/churn_model_comparison.json", "w") as f:
        json.dump({"results": results, "best_model": best_name, "confusion_matrix": cm.tolist()}, f, indent=2)

    # also export test set for downstream SHAP / dashboard demo
    X_test.assign(Churned=y_test.values).to_parquet("data/processed/test_set.parquet", index=False)
    joblib.dump((X_test, y_test), "data/processed/test_arrays.pkl")


if __name__ == "__main__":
    main()
