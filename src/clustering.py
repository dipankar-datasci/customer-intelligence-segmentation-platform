
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

SEGMENTS_PATH = OUTPUT_DIR / "customer_segments.csv"
METRICS_PATH = OUTPUT_DIR / "clustering_metrics.csv"
PROFILES_PATH = OUTPUT_DIR / "cluster_profiles.csv"
MODEL_PATH = MODEL_DIR / "customer_segmentation_model.joblib"

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "ActiveMonths",
]

RANDOM_STATE = 42
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10


# --------------------------------------------------
# 2. Load and validate customer features
# --------------------------------------------------

def load_customer_features():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Customer features not found. "
            "Run `python -m src.feature_engineering` first."
        )

    df = pd.read_csv(
        FEATURES_PATH,
        dtype={"CustomerID": "string"},
    )

    required = ["CustomerID"] + FEATURE_COLUMNS
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be unique.")

    if df[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Feature data contains missing values.")

    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Feature data contains infinite values.")

    if (df[FEATURE_COLUMNS] < 0).any().any():
        raise ValueError(
            "This log1p pipeline expects non-negative features."
        )

    if len(df) <= MAX_CLUSTERS:
        raise ValueError("Not enough customers for the cluster search.")

    print(f"Customers loaded: {len(df):,}")
    print(f"Features used: {FEATURE_COLUMNS}")

    return df


# --------------------------------------------------
# 3. Transform and scale features
# --------------------------------------------------

def prepare_features(df):
    # Log transformation reduces the influence of extreme values.
    X_log = np.log1p(df[FEATURE_COLUMNS].astype(float))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    return X_scaled, scaler


# --------------------------------------------------
# 4. Compare K-Means models
# --------------------------------------------------

def evaluate_kmeans(X_scaled):
    results = []
    models = {}

    for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        print(f"Training K-Means with k={k}...")

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
            max_iter=500,
        )

        labels = model.fit_predict(X_scaled)

        # Silhouette is calculated on a reproducible sample
        # to keep evaluation efficient.
        silhouette = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(3000, len(X_scaled)),
            random_state=RANDOM_STATE,
        )

        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)

        counts = pd.Series(labels).value_counts()
        smallest_cluster_pct = (
            counts.min() / len(labels) * 100
        )

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette,
            "davies_bouldin": db_score,
            "calinski_harabasz": ch_score,
            "smallest_cluster_pct": smallest_cluster_pct,
        })

        models[k] = model

    metrics = pd.DataFrame(results)

    # Use silhouette to rank candidate models, not as the
    # only criterion for final business suitability.
    metrics = metrics.sort_values(
        "silhouette",
        ascending=False,
    ).reset_index(drop=True)

    return metrics, models


# --------------------------------------------------
# 5. Profile customer segments
# --------------------------------------------------

def create_cluster_profiles(segmented_df):
    profiles = (
        segmented_df
        .groupby("Cluster")[FEATURE_COLUMNS]
        .agg(["mean", "median"])
        .round(2)
    )

    # Flatten the multi-level column names.
    profiles.columns = [
        f"{feature}_{stat}"
        for feature, stat in profiles.columns
    ]

    sizes = segmented_df.groupby("Cluster").size()

    profiles.insert(0, "CustomerCount", sizes)
    profiles.insert(
        1,
        "CustomerPercentage",
        (sizes / len(segmented_df) * 100).round(2),
    )

    return profiles.reset_index()


# --------------------------------------------------
# 6. Train selected model and save outputs
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_customer_features()

    X_scaled, scaler = prepare_features(df)

    metrics, models = evaluate_kmeans(X_scaled)

    metrics.to_csv(METRICS_PATH, index=False)

    print("\nModel comparison:")
    print(metrics.round(4).to_string(index=False))

    # Select the best silhouette candidate.
    best_k = int(metrics.iloc[0]["k"])
    best_model = models[best_k]

    print(f"\nBest silhouette candidate: k={best_k}")

    # Assign a cluster to every customer.
    segmented_df = df.copy()
    segmented_df["Cluster"] = best_model.predict(X_scaled)

    segmented_df.to_csv(SEGMENTS_PATH, index=False)

    # Summarize the actual customer behavior in each cluster.
    profiles = create_cluster_profiles(segmented_df)
    profiles.to_csv(PROFILES_PATH, index=False)

    # Save the preprocessing objects and fitted model together.
    joblib.dump(
        {
            "scaler": scaler,
            "model": best_model,
            "feature_columns": FEATURE_COLUMNS,
            "log_transform": "numpy.log1p",
            "selected_k": best_k,
            "random_state": RANDOM_STATE,
        },
        MODEL_PATH,
    )

    print("\nCluster sizes:")
    print(
        segmented_df["Cluster"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCluster profiles:")
    print(profiles.to_string(index=False))

    print("\nSaved files:")
    print(f"Metrics:  {METRICS_PATH}")
    print(f"Segments: {SEGMENTS_PATH}")
    print(f"Profiles: {PROFILES_PATH}")
    print(f"Model:    {MODEL_PATH}")


if __name__ == "__main__":
    main()
