
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEGMENTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customer_segments.csv"
)

PROFILES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cluster_profiles.csv"
)

LABELED_PROFILES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "business_segment_profiles.csv"
)


def load_segments():
    if not SEGMENTS_PATH.exists():
        raise FileNotFoundError(
            "Customer segments not found. "
            "Run `python -m src.clustering` first."
        )

    df = pd.read_csv(
        SEGMENTS_PATH,
        dtype={"CustomerID": "string"},
    )

    required_columns = [
        "CustomerID",
        "Cluster",
        "Recency",
        "Frequency",
        "Monetary",
        "AvgOrderValue",
        "UniqueProducts",
        "ActiveMonths",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    return df


def create_business_labels(df):
    """
    Rank clusters by a composite engagement score.

    Higher frequency, spending, product variety and active months
    increase the score. Higher recency reduces it.

    This is a descriptive ranking, not a validated measure
    of customer loyalty or future churn.
    """

    cluster_stats = (
        df.groupby("Cluster")
        .agg(
            Recency=("Recency", "median"),
            Frequency=("Frequency", "median"),
            Monetary=("Monetary", "median"),
            AvgOrderValue=("AvgOrderValue", "median"),
            UniqueProducts=("UniqueProducts", "median"),
            ActiveMonths=("ActiveMonths", "median"),
            CustomerCount=("CustomerID", "nunique"),
        )
    )

    # Log transformation reduces the influence of extreme values.
    engagement_components = pd.DataFrame(
        {
            "Recency": -np.log1p(cluster_stats["Recency"]),
            "Frequency": np.log1p(cluster_stats["Frequency"]),
            "Monetary": np.log1p(cluster_stats["Monetary"]),
            "UniqueProducts": np.log1p(
                cluster_stats["UniqueProducts"]
            ),
            "ActiveMonths": np.log1p(
                cluster_stats["ActiveMonths"]
            ),
        },
        index=cluster_stats.index,
    )

    # Standardize cluster-level metrics before combining them.
    standard_deviations = engagement_components.std(ddof=0)

    # If a metric is identical across all clusters, its contribution
    # is set to zero instead of dividing by zero.
    standard_deviations = standard_deviations.replace(0, 1)

    standardized = (
        engagement_components - engagement_components.mean()
    ) / standard_deviations

    cluster_stats["EngagementScore"] = standardized.mean(axis=1)

    ordered_clusters = (
        cluster_stats["EngagementScore"]
        .sort_values()
        .index
        .tolist()
    )

    n_clusters = len(ordered_clusters)
    labels = {}

    for position, cluster_id in enumerate(ordered_clusters):
        if n_clusters == 1:
            label = "General Customer Segment"
        elif position == n_clusters - 1:
            label = "Engaged / Higher-Value Customers"
        elif position == 0:
            label = "Infrequent / Less-Engaged Customers"
        else:
            label = f"Developing Customer Segment {position}"

        labels[cluster_id] = label

    cluster_stats["SegmentName"] = pd.Series(labels)

    return cluster_stats, labels


def recommend_action(row):
    """
    Generate a simple recommendation from the segment's
    historical median behavior.
    """

    recency = row["Recency"]
    frequency = row["Frequency"]
    monetary = row["Monetary"]

    if (
        recency >= 180
        and monetary >= 1000
    ):
        return (
            "Prioritize a personalized win-back campaign; "
            "test a relevant offer and measure reactivation."
        )

    if (
        recency <= 60
        and frequency >= 5
    ):
        return (
            "Test loyalty rewards, personalized cross-selling, "
            "and retention offers."
        )

    if recency >= 180:
        return (
            "Test a low-cost re-engagement campaign; "
            "compare response rates before increasing spend."
        )

    if frequency <= 2:
        return (
            "Encourage a second or repeat purchase using "
            "relevant recommendations and follow-up messaging."
        )

    return (
        "Use targeted product recommendations and monitor "
        "repeat purchases, spending, and engagement."
    )


def main():
    df = load_segments()

    cluster_stats, labels = create_business_labels(df)

    # Add business-readable labels to each customer's record.
    df["SegmentName"] = df["Cluster"].map(labels)

    action_map = {
        cluster_id: recommend_action(row)
        for cluster_id, row in cluster_stats.iterrows()
    }

    df["RecommendedAction"] = df["Cluster"].map(action_map)

    # Save the enriched customer table.
    df.to_csv(SEGMENTS_PATH, index=False)

    # Create a business-facing segment summary.
    cluster_stats["RecommendedAction"] = pd.Series(action_map)

    cluster_stats = cluster_stats.reset_index()

    cluster_stats.to_csv(
        LABELED_PROFILES_PATH,
        index=False,
    )

    print("\nBUSINESS SEGMENT SUMMARY")
    print("=" * 80)

    display_columns = [
        "Cluster",
        "SegmentName",
        "CustomerCount",
        "Recency",
        "Frequency",
        "Monetary",
        "ActiveMonths",
        "RecommendedAction",
    ]

    print(
        cluster_stats[display_columns].to_string(index=False)
    )

    print("\nUpdated customer segments:")
    print(SEGMENTS_PATH)

    print("\nBusiness segment profiles:")
    print(LABELED_PROFILES_PATH)


if __name__ == "__main__":
    main()
    