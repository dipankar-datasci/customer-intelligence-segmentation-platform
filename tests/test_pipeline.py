
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "customer_segmentation_model.joblib"


def load_csv(filename):
    path = PROCESSED / filename
    assert path.exists(), f"Missing file: {path}"
    return pd.read_csv(path)


def test_cleaned_transactions_are_valid():
    df = load_csv("cleaned_transactions.csv")

    required = [
        "CustomerID",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "Revenue",
    ]
    assert set(required).issubset(df.columns)
    assert df["CustomerID"].notna().all()
    assert df["InvoiceDate"].notna().all()
    assert (df["Quantity"] > 0).all()
    assert (df["UnitPrice"] > 0).all()


def test_customer_features_are_valid():
    df = load_csv("customer_features.csv")

    required = [
        "CustomerID",
        "Frequency",
        "Monetary",
        "UniqueProducts",
        "Recency",
        "AvgOrderValue",
        "ActiveMonths",
    ]
    assert set(required).issubset(df.columns)
    assert df["CustomerID"].is_unique
    assert df[required[1:]].notna().all().all()
    assert (df[required[1:]] >= 0).all().all()
    assert (df["Frequency"] > 0).all()
    assert (df["ActiveMonths"] > 0).all()


def test_customer_segments_cover_all_customers():
    features = load_csv("customer_features.csv")
    segments = load_csv("customer_segments.csv")

    assert "CustomerID" in segments.columns
    assert "Cluster" in segments.columns
    assert segments["CustomerID"].is_unique

    assert set(features["CustomerID"]) == set(segments["CustomerID"])
    assert segments["Cluster"].notna().all()


def test_business_recommendations_exist():
    segments = load_csv("customer_segments.csv")

    assert "SegmentName" in segments.columns
    assert "RecommendedAction" in segments.columns
    assert segments["SegmentName"].notna().all()
    assert segments["RecommendedAction"].notna().all()


def test_clustering_metrics_are_valid():
    metrics = load_csv("clustering_metrics.csv")

    required = [
        "k",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "smallest_cluster_pct",
    ]
    assert set(required).issubset(metrics.columns)
    assert metrics["k"].is_unique
    assert metrics["silhouette"].between(-1, 1).all()
    assert (metrics["davies_bouldin"] >= 0).all()
    assert (metrics["smallest_cluster_pct"] > 0).all()


def test_saved_model_can_predict_customers():
    assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"

    artifact = joblib.load(MODEL_PATH)

    for key in ["scaler", "model", "feature_columns"]:
        assert key in artifact

    features = load_csv("customer_features.csv")
    columns = artifact["feature_columns"]

    assert set(columns).issubset(features.columns)

    X = np.log1p(features[columns])
    X_scaled = artifact["scaler"].transform(X)
    predictions = artifact["model"].predict(X_scaled)

    assert len(predictions) == len(features)
    assert set(np.unique(predictions)) == set(
        range(artifact["model"].n_clusters)
    )
