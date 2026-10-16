
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

SEGMENTS_PATH = DATA_DIR / "customer_segments.csv"

PROFILES_PATH = DATA_DIR / "business_segment_profiles.csv"

METRICS_PATH = DATA_DIR / "clustering_metrics.csv"


st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_dashboard_data():
    required_files = [
        SEGMENTS_PATH,
        PROFILES_PATH,
        METRICS_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Run the data-processing, feature-engineering, "
                "clustering and business-insights modules first."
            )

    segments = pd.read_csv(
        SEGMENTS_PATH,
        dtype={"CustomerID": "string"},
    )

    profiles = pd.read_csv(PROFILES_PATH)

    metrics = pd.read_csv(METRICS_PATH)

    return segments, profiles, metrics


st.title("Customer Intelligence & Behavioral Segmentation")
st.caption(
    "Retail transaction analytics | Customer segmentation | "
    "Business recommendations"
)

try:
    segments, profiles, metrics = load_dashboard_data()
except (FileNotFoundError, pd.errors.EmptyDataError) as error:
    st.error(str(error))
    st.stop()


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

segment_names = sorted(
    segments["SegmentName"].dropna().unique().tolist()
)

selected_segments = st.sidebar.multiselect(
    "Select customer segments",
    options=segment_names,
    default=segment_names,
)

min_recency = int(segments["Recency"].min())
max_recency = int(segments["Recency"].max())

recency_range = st.sidebar.slider(
    "Recency range (days)",
    min_value=min_recency,
    max_value=max_recency,
    value=(min_recency, max_recency),
)


filtered = segments[
    segments["SegmentName"].isin(selected_segments)
    & segments["Recency"].between(
        recency_range[0],
        recency_range[1],
    )
].copy()


# --------------------------------------------------
# KPI cards
# --------------------------------------------------

total_customers = filtered["CustomerID"].nunique()

total_revenue = filtered["Monetary"].sum()

average_customer_value = (
    filtered["Monetary"].mean()
    if not filtered.empty
    else 0
)

average_recency = (
    filtered["Recency"].mean()
    if not filtered.empty
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers", f"{total_customers:,}")
col2.metric("Historical customer value", f"£{total_revenue:,.2f}")
col3.metric("Average customer value", f"£{average_customer_value:,.2f}")
col4.metric("Average recency", f"{average_recency:.1f} days")


# --------------------------------------------------
# Segment distribution
# --------------------------------------------------

st.header("1. Customer Segment Overview")

if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

segment_counts = (
    filtered.groupby("SegmentName")
    .size()
    .reset_index(name="Customers")
)

left, right = st.columns(2)

with left:
    fig = px.bar(
        segment_counts,
        x="SegmentName",
        y="Customers",
        color="SegmentName",
        title="Customers by Segment",
        text="Customers",
    )

    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.pie(
        segment_counts,
        names="SegmentName",
        values="Customers",
        title="Customer Distribution",
        hole=0.45,
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Customer behavior
# --------------------------------------------------

st.header("2. Customer Behavior Analysis")

left, right = st.columns(2)

with left:
    fig = px.box(
        filtered,
        x="SegmentName",
        y="Monetary",
        color="SegmentName",
        title="Historical Spend by Segment",
        points=False,
    )

    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.scatter(
        filtered,
        x="Recency",
        y="Frequency",
        color="SegmentName",
        size="Monetary",
        hover_data=[
            "CustomerID",
            "AvgOrderValue",
            "ActiveMonths",
        ],
        title="Recency vs Purchase Frequency",
        size_max=24,
        opacity=0.65,
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Business recommendations
# --------------------------------------------------

st.header("3. Business Recommendations")

st.dataframe(
    profiles[
        [
            "SegmentName",
            "CustomerCount",
            "Recency",
            "Frequency",
            "Monetary",
            "ActiveMonths",
            "RecommendedAction",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Model evaluation
# --------------------------------------------------

st.header("4. Clustering Model Evaluation")

st.caption(
    "These are offline clustering metrics. Higher silhouette "
    "and Calinski–Harabasz scores are generally preferable; "
    "lower Davies–Bouldin scores are generally preferable."
)

st.dataframe(
    metrics.sort_values("k"),
    use_container_width=True,
    hide_index=True,
)

fig = px.line(
    metrics.sort_values("k"),
    x="k",
    y="silhouette",
    markers=True,
    title="Silhouette Score by Number of Clusters",
    labels={
        "k": "Number of clusters",
        "silhouette": "Silhouette score",
    },
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Customer lookup and export
# --------------------------------------------------

st.header("5. Customer Explorer")

customer_search = st.text_input(
    "Search by Customer ID",
    placeholder="Enter a customer ID",
)

customer_view = filtered.copy()

if customer_search.strip():
    customer_view = customer_view[
        customer_view["CustomerID"]
        .str.contains(customer_search.strip(), regex=False, na=False)
    ]

display_columns = [
    "CustomerID",
    "SegmentName",
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "ActiveMonths",
    "RecommendedAction",
]

st.dataframe(
    customer_view[display_columns],
    width="stretch",
    hide_index=True,
)

st.download_button(
    label="Download filtered customer segments",
    data=customer_view[display_columns].to_csv(index=False),
    file_name="customer_segments_export.csv",
    mime="text/csv",
)

st.caption(
    "Historical transaction analysis only. Segment membership "
    "does not establish future churn, customer intent, or the "
    "causal impact of a marketing campaign."
)



# --------------------------------------------------
# Model Validation and Robustness
# --------------------------------------------------

st.header("Model Validation & Robustness")

st.write(
    "Compare clustering algorithms and assess whether "
    "K-Means produces consistent customer segments "
    "across different random initializations."
)

metrics_path = (
    "data/processed/algorithm_comparison.csv"
)
stability_path = (
    "data/processed/cluster_stability.csv"
)

if __import__("pathlib").Path(metrics_path).exists():
    comparison = pd.read_csv(metrics_path)

    st.subheader("Algorithm Comparison")

    st.dataframe(
        comparison.sort_values(
            "Silhouette", ascending=False
        ).round(4),
        width="stretch",
        hide_index=True,
    )

    fig = px.bar(
        comparison,
        x="Algorithm",
        y="Silhouette",
        color="k",
        barmode="group",
        title="Silhouette Score by Algorithm and Cluster Count",
        hover_data=[
            "DaviesBouldin",
            "SmallestClusterPct",
        ],
    )

    st.plotly_chart(fig, width="stretch")

else:
    st.warning(
        "Algorithm comparison results not found. "
        "Run: python -m src.algorithm_comparison"
    )

if __import__("pathlib").Path(stability_path).exists():
    stability = pd.read_csv(stability_path)

    st.subheader("Clustering Stability")

    st.dataframe(
        stability.round(4),
        width="stretch",
        hide_index=True,
    )

    fig = px.bar(
        stability,
        x="k",
        y="MeanPairwiseARI",
        title="Consistency Across Random Initializations",
        labels={
            "k": "Number of Clusters",
            "MeanPairwiseARI": "Mean Pairwise ARI",
        },
        hover_data=["MinPairwiseARI"],
    )

    st.plotly_chart(fig, width="stretch")

else:
    st.warning(
        "Stability results not found. "
        "Run: python -m src.cluster_stability"
    )

st.caption(
    "Internal clustering metrics measure separation and compactness. "
    "Stability measures consistency across initializations. "
    "Neither proves that a marketing campaign will improve revenue."
)
