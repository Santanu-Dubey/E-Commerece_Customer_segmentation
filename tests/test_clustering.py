"""
test_clustering.py
------------------
Unit tests for clustering algorithms, silhouette evaluations, and business labeling.
"""

import pytest
import numpy as np
import pandas as pd
from src.synthetic_data import generate_synthetic_transactions
from src.data_prep import DataPreparation
from src.feature_engineering import RFMFeatureEngineer
from src.clustering import CustomerClusterer
from src.profiling import SegmentProfiler


@pytest.fixture
def scaled_feature_data():
    raw_df = generate_synthetic_transactions(num_customers=150, random_seed=42)
    prep = DataPreparation()
    clean_df, _ = prep.clean_transactions(raw_df)
    engineer = RFMFeatureEngineer()
    rfm_df, scaled_matrix, _ = engineer.transform_and_scale(engineer.compute_rfm(clean_df))
    return rfm_df, scaled_matrix


def test_kmeans_clustering(scaled_feature_data):
    rfm_df, scaled_matrix = scaled_feature_data
    clusterer = CustomerClusterer(random_state=42)

    # Test K-Means range evaluation
    metrics_df = clusterer.evaluate_kmeans_range(scaled_matrix, k_range=range(2, 6))
    assert len(metrics_df) == 4
    assert "Silhouette_Score" in metrics_df.columns
    assert "Inertia" in metrics_df.columns

    # Test K-Means fit
    labels, model = clusterer.fit_kmeans(scaled_matrix, k=4)
    assert len(labels) == len(rfm_df)
    assert len(set(labels)) == 4
    assert not np.isnan(labels).any()


def test_agglomerative_and_dbscan(scaled_feature_data):
    rfm_df, scaled_matrix = scaled_feature_data
    clusterer = CustomerClusterer(random_state=42)

    # Test Agglomerative
    agg_labels, _ = clusterer.fit_agglomerative(scaled_matrix, n_clusters=4)
    assert len(agg_labels) == len(rfm_df)
    assert len(set(agg_labels)) == 4

    # Test DBSCAN
    db_labels, _ = clusterer.fit_dbscan(scaled_matrix, eps=0.7, min_samples=5)
    assert len(db_labels) == len(rfm_df)


def test_business_profiling(scaled_feature_data):
    rfm_df, scaled_matrix = scaled_feature_data
    clusterer = CustomerClusterer(random_state=42)
    labels, _ = clusterer.fit_kmeans(scaled_matrix, k=4)
    rfm_df["Cluster"] = labels

    profiler = SegmentProfiler(rfm_df, cluster_col="Cluster")
    df_labeled, cluster_map = profiler.assign_business_labels()

    assert "Segment" in df_labeled.columns
    assert df_labeled["Segment"].isna().sum() == 0

    summary_df = profiler.generate_segment_summary()
    assert len(summary_df) == 4
    assert "Customer_Share_Pct" in summary_df.columns
    assert "Revenue_Share_Pct" in summary_df.columns
    assert np.isclose(summary_df["Customer_Share_Pct"].sum(), 100.0, atol=0.2)
