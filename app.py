"""
app.py
------
Interactive Streamlit Web Application for E-Commerce Customer Segmentation & Marketing Intelligence.
Allows marketing teams and executives to explore segments, inspect RFM profiles,
simulate campaigns, and download customer segment lists.
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page layout
st.set_page_config(
    page_title="E-Commerce Customer Segmentation & Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_segmentation_data():
    """Loads segmented customer data and summary tables."""
    data_path = "data/customer_rfm_segmented.csv"
    summary_path = "outputs/tables/cluster_summary_metrics.csv"
    playbook_path = "outputs/tables/segment_business_profiles.csv"
    comparison_path = "outputs/tables/model_comparison_metrics.csv"

    if not os.path.exists(data_path):
        from main import run_pipeline
        run_pipeline()

    df = pd.read_csv(data_path)
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    playbook_df = pd.read_csv(playbook_path) if os.path.exists(playbook_path) else None
    comparison_df = pd.read_csv(comparison_path) if os.path.exists(comparison_path) else None

    return df, summary_df, playbook_df, comparison_df


def main():
    st.markdown('<div class="main-header">🛍️ E-Commerce Customer Segmentation & Marketing Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered RFM Behavioral Segmentation, Cluster Analytics, & Strategic Campaign Playbooks</div>', unsafe_allow_html=True)

    df, summary_df, playbook_df, comparison_df = load_segmentation_data()

    # Sidebar Filters
    st.sidebar.header("🎯 Segmentation Filters")
    all_segments = sorted(df["Segment"].unique())
    selected_segments = st.sidebar.multiselect("Select Segments to View:", all_segments, default=all_segments)

    min_spend, max_spend = float(df["Monetary"].min()), float(df["Monetary"].max())
    spend_range = st.sidebar.slider("Total Spend Range ($):", min_spend, max_spend, (min_spend, max_spend))

    filtered_df = df[
        (df["Segment"].isin(selected_segments)) &
        (df["Monetary"] >= spend_range[0]) &
        (df["Monetary"] <= spend_range[1])
    ]

    # Top KPI Cards
    total_cust = len(filtered_df)
    total_rev = filtered_df["Monetary"].sum()
    avg_aov = filtered_df["AvgOrderValue"].mean()
    avg_rec = filtered_df["Recency"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers in View", f"{total_cust:,}", f"{len(df):,} Total")
    with col2:
        st.metric("Total Revenue in View", f"${total_rev:,.2f}", f"${df['Monetary'].sum():,.2f} Total")
    with col3:
        st.metric("Average Order Value (AOV)", f"${avg_aov:.2f}")
    with col4:
        st.metric("Average Recency", f"{avg_rec:.1f} days")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Segment Overview",
        "🌌 2D & 3D Visual Cluster Explorer",
        "🎯 Marketing Campaign Playbook",
        "🔍 Customer Lookup & Table",
        "🧪 Model Comparison & Diagnostics",
    ])

    # TAB 1: OVERVIEW
    with tab1:
        st.subheader("Segment Distribution & Pareto Revenue Dynamics")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_cust = px.pie(
                filtered_df,
                names="Segment",
                title="Customer Volume Breakdown (%)",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_cust.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_cust, use_container_width=True)

        with col_b:
            rev_by_seg = filtered_df.groupby("Segment")["Monetary"].sum().reset_index()
            fig_rev = px.pie(
                rev_by_seg,
                values="Monetary",
                names="Segment",
                title="Revenue Contribution Breakdown (%)",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_rev.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_rev, use_container_width=True)

        st.subheader("Segment Metrics Comparison")
        if summary_df is not None:
            st.dataframe(
                summary_df.style.format({
                    "Total_Revenue": "${:,.2f}",
                    "Recency_Mean": "{:.1f} d",
                    "Frequency_Mean": "{:.1f} ord",
                    "Monetary_Mean": "${:,.2f}",
                    "AOV_Mean": "${:,.2f}",
                    "Customer_Share_Pct": "{:.1f}%",
                    "Revenue_Share_Pct": "{:.1f}%",
                }),
                use_container_width=True,
            )

    # TAB 2: VISUAL CLUSTER EXPLORER
    with tab2:
        st.subheader("2D PCA & 3D Interactive Spatial Projection")
        view_mode = st.radio("Select View:", ["3D Interactive RFM Space", "2D Principal Component Analysis (PCA)"], horizontal=True)

        if view_mode == "3D Interactive RFM Space":
            fig_3d = px.scatter_3d(
                filtered_df.sample(min(2500, len(filtered_df)), random_state=42) if len(filtered_df) > 2500 else filtered_df,
                x="Recency",
                y="Frequency",
                z="Monetary",
                color="Segment",
                log_x=True,
                log_y=True,
                log_z=True,
                hover_data=["CustomerID", "AvgOrderValue"],
                opacity=0.75,
                height=650,
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            fig_2d = px.scatter(
                filtered_df,
                x="PCA1",
                y="PCA2",
                color="Segment",
                hover_data=["CustomerID", "Recency", "Frequency", "Monetary"],
                title="2D PCA Latent Space Projection",
                height=600,
            )
            st.plotly_chart(fig_2d, use_container_width=True)

    # TAB 3: MARKETING PLAYBOOK
    with tab3:
        st.subheader("Strategic Marketing Action Playbook")
        if playbook_df is not None:
            for _, row in playbook_df.iterrows():
                with st.expander(f"📌 **{row['Segment_Name']}** ({row['Customer_Share_Pct']} Customers | {row['Revenue_Share_Pct']} Revenue)", expanded=True):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(f"**Priority:** `{row['Priority']}`")
                        st.markdown(f"**Avg Spend:** `{row['Avg_Monetary_Spend']}`")
                        st.markdown(f"**Avg Frequency:** `{row['Avg_Frequency_Orders']}` orders")
                        st.markdown(f"**Avg Recency:** `{row['Avg_Recency_Days']}` days")
                        st.markdown(f"**Channel:** `{row['Preferred_Channel']}`")
                    with c2:
                        st.markdown(f"**Customer Archetype:** {row['Business_Description']}")
                        st.markdown(f"**Action Plan:** :blue[**{row['Recommended_Action']}**]")
                        st.markdown(f"**Target KPIs:** `{row['Target_KPIs']}`")

    # TAB 4: CUSTOMER LOOKUP & DATA
    with tab4:
        st.subheader("Customer Level Segmentation Records")
        search_id = st.text_input("🔍 Search by Customer ID:")
        view_table = filtered_df
        if search_id.strip():
            view_table = view_table[view_table["CustomerID"].astype(str).str.contains(search_id.strip())]

        st.dataframe(
            view_table[[
                "CustomerID", "Segment", "Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays", "UniqueItemsPurchased"
            ]].style.format({
                "Monetary": "${:,.2f}",
                "AvgOrderValue": "${:,.2f}",
                "Recency": "{:.0f} days",
                "Frequency": "{:.0f}",
                "TenureDays": "{:.0f} days",
            }),
            use_container_width=True,
        )

        csv = view_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Customer Segment CSV",
            data=csv,
            file_name="segmented_customers_export.csv",
            mime="text/csv",
        )

    # TAB 5: MODEL COMPARISON
    with tab5:
        st.subheader("Clustering Algorithm Benchmark")
        if comparison_df is not None:
            st.dataframe(comparison_df, use_container_width=True)
            st.info("""
            **Metric Guidance:**
            - **Silhouette Score**: Ranges from -1 to +1 (higher is better). Measures cluster cohesion vs separation.
            - **Davies-Bouldin Index**: Lower is better (0 is ideal). Measures average similarity of clusters.
            - **Calinski-Harabasz Index**: Higher is better. Ratio of between-cluster dispersion to within-cluster dispersion.
            """)


if __name__ == "__main__":
    main()
