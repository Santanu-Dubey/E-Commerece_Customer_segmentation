"""
test_data_prep.py
-----------------
Unit tests for data preparation and hygiene pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.synthetic_data import generate_synthetic_transactions
from src.data_prep import DataPreparation


@pytest.fixture
def sample_raw_df():
    """Generates a small synthetic dataframe for fast testing."""
    return generate_synthetic_transactions(num_customers=50, random_seed=42)


def test_synthetic_data_generation(sample_raw_df):
    """Verifies that synthetic data generation produces expected columns and rows."""
    assert len(sample_raw_df) > 0
    expected_cols = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
    assert expected_cols.issubset(set(sample_raw_df.columns))
    assert sample_raw_df["CustomerID"].nunique() > 0


def test_data_cleaning_hygiene(sample_raw_df):
    """Verifies that data cleaning eliminates invalid entries, cancellations, and NaNs."""
    prep = DataPreparation()
    clean_df, stats = prep.clean_transactions(sample_raw_df)

    # 1. No NaNs in CustomerID
    assert clean_df["CustomerID"].isna().sum() == 0

    # 2. No non-positive quantities or unit prices
    assert (clean_df["Quantity"] <= 0).sum() == 0
    assert (clean_df["UnitPrice"] <= 0).sum() == 0

    # 3. No cancellations (invoices starting with 'C')
    assert clean_df["InvoiceNo"].str.startswith("C").sum() == 0

    # 4. TotalSpend correctly computed
    assert "TotalSpend" in clean_df.columns
    np.testing.assert_allclose(clean_df["TotalSpend"], clean_df["Quantity"] * clean_df["UnitPrice"])

    # 5. Diagnostic stats dictionary structure
    assert "initial_rows" in stats
    assert "final_clean_rows" in stats
    assert stats["final_clean_rows"] <= stats["initial_rows"]
