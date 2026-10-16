
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEANED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.csv"
)

FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
)


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert transaction-level data into one row per customer.
    """

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Use the day after the final observed transaction as the
    # reference date for calculating customer recency.
    reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    # Calculate customer-level RFM and behavioral features.
    features = (
        df.groupby("CustomerID")
        .agg(
            LastPurchaseDate=("InvoiceDate", "max"),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
            UniqueProducts=("StockCode", "nunique"),
        )
        .reset_index()
    )

    # Recency: number of days since the most recent purchase.
    features["Recency"] = (
        reference_date - features["LastPurchaseDate"].dt.normalize()
    ).dt.days

    # Average order value: total spend / distinct invoices.
    features["AvgOrderValue"] = (
        features["Monetary"] / features["Frequency"]
    )

    # Number of distinct calendar months in which a customer purchased.
    active_months = (
        df.assign(
            PurchaseMonth=df["InvoiceDate"].dt.to_period("M")
        )
        .groupby("CustomerID")["PurchaseMonth"]
        .nunique()
        .rename("ActiveMonths")
    )

    features = features.merge(
        active_months,
        on="CustomerID",
        how="left",
    )

    # Keep the reference date in memory for documentation,
    # but it is not a model feature.
    features.attrs["reference_date"] = str(reference_date.date())

    # Move the latest purchase date out of the modeling columns.
    features = features.drop(columns=["LastPurchaseDate"])

    # Ensure the feature table has one row per customer.
    if features["CustomerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs found in feature table.")

    return features


def main() -> None:
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned transaction data not found. "
            "Run `python -m src.data_processing` first."
        )

    print("Loading cleaned transactions...")

    df = pd.read_csv(
        CLEANED_DATA_PATH,
        dtype={"CustomerID": "string"},
        parse_dates=["InvoiceDate"],
    )

    print(f"Transaction rows: {len(df):,}")

    features = create_customer_features(df)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(FEATURES_PATH, index=False)

    print("\nCustomer feature table created.")
    print(f"Number of customers: {len(features):,}")
    print(f"Number of columns: {len(features.columns)}")
    print("\nFeature summary:")
    print(features.describe().round(2).to_string())

    print(f"\nSaved to: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
