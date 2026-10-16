
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "customer_features.csv"
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "algorithm_comparison.csv"
)

FEATURES = [
    "Frequency",
    "Monetary",
    "UniqueProducts",
    "Recency",
    "AvgOrderValue",
    "ActiveMonths",
]


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()

    if X.isna().any().any() or (X < 0).any().any():
        raise ValueError("Features must be non-negative and complete.")

    X_log = np.log1p(X)
    X_scaled = StandardScaler().fit_transform(X_log)

    results = []

    for k in [2, 3, 4]:
        algorithms = {
            "KMeans": KMeans(
                n_clusters=k,
                random_state=42,
                n_init=20,
                max_iter=500,
            ),
            "Agglomerative": AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
            ),
            "GaussianMixture": GaussianMixture(
                n_components=k,
                random_state=42,
                n_init=3,
                reg_covar=1e-5,
            ),
        }

        for name, algorithm in algorithms.items():
            if name == "GaussianMixture":
                algorithm.fit(X_scaled)
                labels = algorithm.predict(X_scaled)
            else:
                labels = algorithm.fit_predict(X_scaled)

            sizes = pd.Series(labels).value_counts()

            results.append({
                "Algorithm": name,
                "k": k,
                "Silhouette": silhouette_score(
                    X_scaled,
                    labels,
                    sample_size=min(3000, len(df)),
                    random_state=42,
                ),
                "DaviesBouldin": davies_bouldin_score(
                    X_scaled, labels
                ),
                "CalinskiHarabasz": calinski_harabasz_score(
                    X_scaled, labels
                ),
                "SmallestClusterPct": (
                    sizes.min() / len(df) * 100
                ),
            })

            print(
                f"{name:18} k={k} | "
                f"Silhouette={results[-1]['Silhouette']:.4f} | "
                f"DB={results[-1]['DaviesBouldin']:.4f} | "
                f"Smallest={results[-1]['SmallestClusterPct']:.2f}%"
            )

    results_df = pd.DataFrame(results).sort_values(
        ["Silhouette", "DaviesBouldin"],
        ascending=[False, True],
    )

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nALGORITHM COMPARISON")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
