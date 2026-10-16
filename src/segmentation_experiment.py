
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "customer_features.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Customer-level behavioral features
FEATURES = [
    "Frequency",
    "Monetary",
    "UniqueProducts",
    "Recency",
    "AvgOrderValue",
    "ActiveMonths",
]


def main():
    print("Loading customer features...")
    df = pd.read_csv(DATA_PATH)

    required_columns = ["CustomerID"] + FEATURES
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must contain one row per customer.")

    X = df[FEATURES].copy()

    if X.isna().any().any():
        raise ValueError("Features contain missing values.")

    if (X < 0).any().any():
        raise ValueError("Log transformation requires non-negative features.")

    # Apply the same preprocessing approach as the baseline model
    X_log = np.log1p(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    metrics = []
    business_profiles = []

    print(f"Customers loaded: {len(df):,}")
    print("Testing cluster counts from 2 to 6...\n")

    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
            max_iter=500,
        )

        labels = model.fit_predict(X_scaled)
        cluster_sizes = pd.Series(labels).value_counts()

        # Clustering quality metrics
        metrics.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(
                X_scaled,
                labels,
                sample_size=min(3000, len(df)),
                random_state=42,
            ),
            "davies_bouldin": davies_bouldin_score(
                X_scaled, labels
            ),
            "calinski_harabasz": calinski_harabasz_score(
                X_scaled, labels
            ),
            "smallest_cluster_pct": (
                cluster_sizes.min() / len(df) * 100
            ),
        })

        # Business profiles use original, interpretable units
        profile_data = df[FEATURES].copy()
        profile_data["Cluster"] = labels

        for cluster_id, group in profile_data.groupby("Cluster"):
            customer_count = len(group)

            profile = {
                "k": k,
                "Cluster": int(cluster_id),
                "CustomerCount": customer_count,
                "CustomerSharePct": (
                    customer_count / len(df) * 100
                ),
            }

            for feature in FEATURES:
                profile[f"{feature}_Median"] = group[
                    feature
                ].median()

            business_profiles.append(profile)

        print(
            f"k={k}: "
            f"silhouette="
            f"{metrics[-1]['silhouette']:.4f}, "
            f"Davies-Bouldin="
            f"{metrics[-1]['davies_bouldin']:.4f}, "
            f"smallest cluster="
            f"{metrics[-1]['smallest_cluster_pct']:.2f}%"
        )

    # Save reports without changing the existing model
    metrics_df = pd.DataFrame(metrics).sort_values(
        "k"
    )

    profiles_df = pd.DataFrame(business_profiles).sort_values(
        ["k", "Cluster"]
    )

    metrics_path = (
        OUTPUT_DIR / "segmentation_experiment_metrics.csv"
    )
    profiles_path = (
        OUTPUT_DIR / "segmentation_business_comparison.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)
    profiles_df.to_csv(profiles_path, index=False)

    print("\nQUALITY METRICS")
    print(metrics_df.round(4).to_string(index=False))

    print("\nBUSINESS PROFILES")
    print(profiles_df.round(2).to_string(index=False))

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved profiles: {profiles_path}")
    print("\nExperiment completed. Existing model unchanged.")


if __name__ == "__main__":
    main()
