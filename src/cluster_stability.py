
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "customer_features.csv"

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
        raise ValueError("Features must be complete and non-negative.")

    X_scaled = StandardScaler().fit_transform(np.log1p(X))

    seeds = [0, 7, 21, 42, 84, 123, 202, 314, 512, 999]
    results = []

    for k in [2, 3, 4]:
        all_labels = []
        silhouettes = []

        for seed in seeds:
            model = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=20,
                max_iter=500,
            )

            labels = model.fit_predict(X_scaled)
            all_labels.append(labels)

            silhouettes.append(
                silhouette_score(
                    X_scaled,
                    labels,
                    sample_size=min(3000, len(df)),
                    random_state=42,
                )
            )

        # ARI compares partitions without relying on cluster IDs
        ari_scores = [
            adjusted_rand_score(all_labels[i], all_labels[j])
            for i, j in combinations(range(len(seeds)), 2)
        ]

        results.append({
            "k": k,
            "Runs": len(seeds),
            "MeanSilhouette": np.mean(silhouettes),
            "MinSilhouette": np.min(silhouettes),
            "MaxSilhouette": np.max(silhouettes),
            "MeanPairwiseARI": np.mean(ari_scores),
            "MinPairwiseARI": np.min(ari_scores),
        })

    results_df = pd.DataFrame(results)

    output_path = (
        ROOT / "data" / "processed" / "cluster_stability.csv"
    )
    results_df.to_csv(output_path, index=False)

    print("\nCLUSTER STABILITY RESULTS")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
