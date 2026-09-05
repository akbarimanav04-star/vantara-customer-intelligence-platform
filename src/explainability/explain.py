import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    pre = joblib.load("models_artifacts/preprocessor.pkl")
    model = joblib.load("models_artifacts/churn_model.pkl")
    X_test, y_test = joblib.load("data/processed/test_arrays.pkl")

    X_test_t = pre.transform(X_test)
    feature_names = pre.get_feature_names_out()
    X_test_df = pd.DataFrame(X_test_t, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)
    # RandomForest binary classifier -> shap_values may be a list [class0, class1] or array
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    plt.figure()
    shap.summary_plot(sv, X_test_df, show=False)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png", dpi=140)
    plt.close()

    # global mean |SHAP| ranking
    mean_abs = np.abs(sv).mean(axis=0)
    ranking = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    ranking.to_csv("reports/shap_feature_importance.csv")
    print("Top drivers of churn:\n", ranking.head(8))

    # a plain-language explanation for 3 representative customers
    proba = model.predict_proba(X_test_t)[:, 1]
    order = np.argsort(proba)
    low_idx, mid_idx, high_idx = order[5], order[len(order)//2], order[-5]

    def explain_customer(idx):
        row_shap = sv[idx]
        top_feats = pd.Series(row_shap, index=feature_names).abs().sort_values(ascending=False).head(3).index
        parts = []
        for f in top_feats:
            val = X_test_df.iloc[idx][f]
            direction = "increasing" if pd.Series(row_shap, index=feature_names)[f] > 0 else "decreasing"
            parts.append(f"{f} ({direction} churn risk, value={val:.2f})")
        return f"Churn probability: {proba[idx]:.2%}. Main drivers: " + "; ".join(parts)

    examples = {
        "low_risk_example": explain_customer(low_idx),
        "borderline_example": explain_customer(mid_idx),
        "high_risk_example": explain_customer(high_idx),
    }
    for k, v in examples.items():
        print(f"\n{k}: {v}")

    import json
    with open("reports/shap_customer_examples.json", "w") as f:
        json.dump(examples, f, indent=2)


if __name__ == "__main__":
    main()
