import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay

def main():
    pre = joblib.load("models_artifacts/preprocessor.pkl")
    model = joblib.load("models_artifacts/churn_model.pkl")
    X_test, y_test = joblib.load("data/processed/test_arrays.pkl")
    X_test_t = pre.transform(X_test)
    feature_names = list(pre.get_feature_names_out())

    # ---- LIME on the same high-risk customer used for the SHAP example ----
    proba = model.predict_proba(X_test_t)[:, 1]
    order = np.argsort(proba)
    high_idx = order[-5]

    explainer = LimeTabularExplainer(
        training_data=X_test_t,
        feature_names=feature_names,
        class_names=["Retained", "Churned"],
        mode="classification",
        random_state=42,
    )
    exp = explainer.explain_instance(X_test_t[high_idx], model.predict_proba, num_features=6)
    lime_list = exp.as_list()
    print("LIME explanation for the high-risk customer:")
    for feat, weight in lime_list:
        print(f"  {feat}: {weight:+.4f}")

    with open("reports/lime_high_risk_example.json", "w") as f:
        json.dump({"churn_probability": float(proba[high_idx]), "lime_explanation": lime_list}, f, indent=2)

    # ---- Partial Dependence Plots for the top drivers ----
    top_features_idx = [feature_names.index("num__Recency"), feature_names.index("num__EngagementScore")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    PartialDependenceDisplay.from_estimator(model, X_test_t, top_features_idx, feature_names=feature_names, ax=ax)
    plt.tight_layout()
    plt.savefig("reports/figures/pdp_top_features.png", dpi=140)
    plt.close()
    print("PDP saved to reports/figures/pdp_top_features.png")


if __name__ == "__main__":
    main()
