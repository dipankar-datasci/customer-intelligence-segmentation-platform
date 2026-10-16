# Customer Intelligence & Behavioral Segmentation Platform

An end-to-end customer analytics project that uses unsupervised machine learning to segment retail customers based on purchasing behavior and translate those segments into actionable business strategies.

## Project Overview

Retail businesses need to understand how customers differ in purchase frequency, spending, product variety, and recency. This project builds a customer segmentation pipeline that transforms transaction-level retail data into customer-level behavioral profiles.

The platform includes data cleaning, feature engineering, clustering, model evaluation, business recommendations, and an interactive Streamlit dashboard.

## Business Objectives

- Identify distinct customer purchasing patterns.
- Compare clustering algorithms using internal validation metrics.
- Evaluate whether the segmentation is stable across different random initializations.
- Translate customer profiles into potential retention, loyalty, and re-engagement strategies.
- Provide an interactive dashboard for exploring segments and customer-level data.

## Dataset

**Online Retail II** — UCI Machine Learning Repository

Dataset source: https://archive.ics.uci.edu/dataset/502/online+retail+ii

The dataset contains historical online retail transactions covering two periods: December 2009–December 2010 and December 2010–December 2011.

The project combines both sheets and applies data-quality filters before generating customer features.

The original dataset is not included in this repository. Download it from the source above and place the Excel file at `data/raw/online_retail_II.xlsx`.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Plotly and Streamlit
- Joblib
- Pytest

## Pipeline

1. **Data processing:** Combine the source sheets, standardize column names, remove duplicate records, and filter cancellations, missing customer identifiers, and invalid quantities or prices.
2. **Feature engineering:** Aggregate transactions into customer-level metrics, including recency, frequency, monetary value, unique products, average order value, and active months.
3. **Preprocessing:** Apply log transformations and standard scaling to reduce skew and place features on comparable scales.
4. **Clustering:** Train K-Means models for different cluster counts.
5. **Model comparison:** Compare K-Means, agglomerative clustering, and Gaussian mixture models using internal clustering metrics.
6. **Stability analysis:** Repeat K-Means across multiple random seeds and compare assignments using the Adjusted Rand Index.
7. **Business insights:** Create customer profiles and proposed actions for engagement, retention, and re-engagement.
8. **Dashboard:** Explore segments, customer metrics, visualizations, and business recommendations using Streamlit.

## Results

After data cleaning, the pipeline retained **793,609 transactions** and produced customer-level features for **5,878 customers**.

The initial K-Means baseline selected two clusters based on silhouette score:

| Metric | Result |
|---|---:|
| Number of customers | 5,878 |
| Selected cluster count | 2 |
| Silhouette score | 0.3778 |
| Davies–Bouldin index | 1.0092 |
| Calinski–Harabasz score | 4719.9834 |
| Mean pairwise ARI across tested seeds | 0.9901 |

The two resulting segments broadly represent more engaged, higher-value customers and less engaged, infrequent customers.

These are historical behavioral profiles, not predictions of future customer behavior. Cluster labels are descriptive and may change when the data or modeling approach changes.

## Project Structure

```text
Customer_Intelligence_Platform/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── business_insights.py
│   ├── segmentation_experiment.py
│   ├── algorithm_comparison.py
│   └── cluster_stability.py
├── tests/
│   └── test_pipeline.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository and navigate into the project directory:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Customer_Intelligence_Platform
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the dataset from the UCI source and place it at:

`data/raw/online_retail_II.xlsx`

## Run the Pipeline

Run these commands from the project root in the following order:

```powershell
python -m src.data_processing
python -m src.feature_engineering
python -m src.clustering
python -m src.business_insights
python -m src.algorithm_comparison
python -m src.cluster_stability
```

Run the automated tests:

```powershell
python -m pytest tests/ -v
```

Launch the dashboard after generating the required outputs:

```powershell
streamlit run app.py
```

## Testing

The current test suite contains six automated tests covering data validity, customer features, segment coverage, business recommendations, clustering metrics, and saved-model predictions.

All six tests passed in the current development environment.

## Limitations and Future Improvements

- The data is historical, so recency values refer to the dataset's final transaction date rather than the present day.
- Clustering metrics assess internal structure and do not prove that the segments improve business outcomes.
- Marketing recommendations are hypotheses that require controlled experiments to validate.
- Future work could evaluate additional clustering methods, alternative feature sets, segment drift, and campaign-response outcomes.

## Author

Developed as an end-to-end customer analytics and machine learning project for practical business intelligence and data science applications.
