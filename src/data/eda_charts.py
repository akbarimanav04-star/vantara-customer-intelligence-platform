import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def main():
    df = pd.read_parquet("data/processed/customer_segments.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df["Recency"], bins=40, ax=axes[0], color="#2E6F95")
    axes[0].set_title("Recency distribution (days)")
    sns.histplot(df["Frequency"], bins=40, ax=axes[1], color="#2E6F95")
    axes[1].set_title("Frequency distribution (orders)")
    sns.histplot(df["Monetary"].clip(upper=df["Monetary"].quantile(0.95)), bins=40, ax=axes[2], color="#2E6F95")
    axes[2].set_title("Monetary distribution (clipped @95th pct)")
    plt.tight_layout()
    plt.savefig("reports/figures/rfm_distributions.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 5))
    seg_counts = df["Segment"].value_counts()
    sns.barplot(x=seg_counts.values, y=seg_counts.index, hue=seg_counts.index, palette="viridis", legend=False)
    plt.title("Customers per segment")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig("reports/figures/segment_counts.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="Recency", y="Monetary", hue="Segment", alpha=0.6, palette="tab10")
    plt.yscale("log")
    plt.title("Recency vs Monetary by segment")
    plt.tight_layout()
    plt.savefig("reports/figures/recency_vs_monetary.png", dpi=140)
    plt.close()

    print("Charts saved to reports/figures/")


if __name__ == "__main__":
    main()
