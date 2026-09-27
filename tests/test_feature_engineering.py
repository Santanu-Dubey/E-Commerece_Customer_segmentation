"""
test_feature_engineering.py
---------------------------
Unit tests for RFM feature engineering, log transformations, and scaling.
"""

import pytest
import numpy as np
import pandas as pd
from src.synthetic_data import generate_synthetic_transactions
from src.data_prep import DataPreparation
from src.feature_engineering import RFMFeatureEngineer


@pytest.fixture
def clean_test_df():
    raw_df = generate_synthetic_transactions(num_customers=100, random_seed=42)
    prep = DataPreparation()
    clean_df, _ = prep.clean_transactions(raw_df)
    return clean_df


def test_rfm_calculation(clean_test_df):
    engineer = RFMFeatureEngineer()
    rfm = engineer.compute_rfm(clean_test_df)

    # Validate customer count
    assert len(rfm) == clean_test_df["CustomerID"].nunique()

    # Validate essential RFM columns
    required_cols = {"CustomerID", "Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays"}
    assert required_cols.issubset(set(rfm.columns))

    # All recency >= 1, frequency >= 1, monetary > 0
    assert (rfm["Recency"] < 1).sum() == 0
    assert (rfm["Frequency"] < 1).sum() == 0
    assert (rfm["Monetary"] <= 0).sum() == 0
    assert (rfm["AvgOrderValue"] <= 0).sum() == 0


def test_rfm_transformation_and_scaling(clean_test_df):
    engineer = RFMFeatureEngineer()
    rfm_raw = engineer.compute_rfm(clean_test_df)
    rfm_df, scaled_matrix, skew_df = engineer.transform_and_scale(rfm_raw)

    # Validate matrix dimensions
    assert scaled_matrix.shape[0] == len(rfm_df)
    assert scaled_matrix.shape[1] == 4  # Recency, Frequency, Monetary, AvgOrderValue

    # Validate StandardScaler properties (mean approx 0, std approx 1)
    means = np.mean(scaled_matrix, axis=0)
    stds = np.std(scaled_matrix, axis=0)
    np.testing.assert_allclose(means, np.zeros_like(means), atol=1e-5)
    np.testing.assert_allclose(stds, np.ones_like(stds), atol=1e-5)

    # Validate skewness reduction report
    assert len(skew_df) == 4
    assert "Raw_Skewness" in skew_df.columns
    assert "Log_Skewness" in skew_df.columns
