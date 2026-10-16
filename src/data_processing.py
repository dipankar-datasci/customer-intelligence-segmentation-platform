
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_transactions.csv"


def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load and combine all sheets from the Online Retail II workbook.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Place online_retail_II.xlsx inside data/raw/."
        )

    print("Loading Excel workbook...")

    # Read every worksheet, not just the first one.
    sheets = pd.read_excel(file_path, sheet_name=None)

    frames = []

    for sheet_name, sheet_df in sheets.items():
        print(f"Loaded sheet: {sheet_name} | Rows: {len(sheet_df):,}")

        sheet_df = sheet_df.copy()
        sheet_df["SourceSheet"] = sheet_name

        frames.append(sheet_df)

    if not frames:
        raise ValueError("The Excel workbook contains no worksheets.")

    df = pd.concat(frames, ignore_index=True)

    # Standardize column names for easier coding.
    df.columns = df.columns.str.strip()

    df = df.rename(
        columns={
            "Customer ID": "CustomerID",
            "Price": "UnitPrice",
        }
    )

    required_columns = {
        "Invoice",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert columns to their expected data types.
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"], errors="coerce"
    )

    # Customer IDs may be missing in the original data.
    df["CustomerID"] = df["CustomerID"].astype("string").str.strip()

    # Calculate line-level transaction value.
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    print(f"\nCombined dataset shape: {df.shape}")

    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Display an initial data-quality report.
    """

    print("\n" + "=" * 60)
    print("RAW DATA QUALITY REPORT")
    print("=" * 60)

    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nDuplicate rows:", df.duplicated().sum())

    cancelled = (
        df["Invoice"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.startswith("C", na=False)
    )

    print("Cancellation invoice rows:", int(cancelled.sum()))

    print(
        "Rows with non-positive quantity:",
        int((df["Quantity"] <= 0).sum()),
    )

    print(
        "Rows with non-positive unit price:",
        int((df["UnitPrice"] <= 0).sum()),
    )

    print(
        "Rows without customer ID:",
        int(df["CustomerID"].isna().sum()
            + df["CustomerID"].eq("").sum()),
    )

    print("\nFirst five rows:")
    print(df.head().to_string())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a customer-analysis dataset by removing unusable
    and non-purchase transaction rows.
    """

    cleaned = df.copy()

    initial_rows = len(cleaned)

    # Remove exact duplicate records.
    cleaned = cleaned.drop_duplicates()

    # Exclude cancellation invoices.
    invoice_text = (
        cleaned["Invoice"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    cleaned = cleaned.loc[
        ~invoice_text.str.startswith("C", na=False)
    ].copy()

    # Keep rows with valid customer, date, quantity and price.
    cleaned = cleaned.dropna(
        subset=[
            "CustomerID",
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
        ]
    )

    cleaned = cleaned.loc[
        (cleaned["CustomerID"] != "")
        & (cleaned["Quantity"] > 0)
        & (cleaned["UnitPrice"] > 0)
    ].copy()

    # Recalculate revenue after filtering.
    cleaned["Revenue"] = (
        cleaned["Quantity"] * cleaned["UnitPrice"]
    )

    # Sort transactions chronologically.
    cleaned = cleaned.sort_values("InvoiceDate")

    cleaned = cleaned.reset_index(drop=True)

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)

    print(f"Rows before cleaning: {initial_rows:,}")
    print(f"Rows after cleaning:  {len(cleaned):,}")
    print(f"Rows removed:         {initial_rows - len(cleaned):,}")
    print(
        f"Percentage retained:  "
        f"{len(cleaned) / initial_rows * 100:.2f}%"
        if initial_rows
        else "Percentage retained: N/A"
    )

    return cleaned


def save_cleaned_data(df: pd.DataFrame) -> None:
    """
    Save the cleaned transaction dataset.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"\nCleaned data saved to:\n{CLEANED_DATA_PATH}")


def main() -> None:
    """
    Run the initial data-processing pipeline.
    """

    raw_df = load_raw_data()

    inspect_data(raw_df)

    cleaned_df = clean_data(raw_df)

    save_cleaned_data(cleaned_df)


if __name__ == "__main__":
    main()
