"""
feature_engineering.py
-----------------------
Transforms cleaned transaction records into customer-level RFM (Recency, Frequency, Monetary)
and behavioral features, applies mathematical transformations (log1p) to remedy heavy skewness,
and normalizes features using StandardScaler.
"""

from __future__ import annotations
import logging
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RFMFeatureEngineer:
    """
    Computes customer-level RFM metrics, performs skewness corrections,
    and applies standard scaling.
    """

    def __init__(self, reference_date: pd.Timestamp | None = None):
        self.reference_date = reference_date
        self.scaler = StandardScaler()
        self.feature_cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue"]
        self.log_feature_cols = [f"{col}_log" for col in self.feature_cols]

    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Recency, Frequency, Monetary, and Average Order Value per customer.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned transaction records.

        Returns
        -------
        pd.DataFrame
            Customer-level summary dataframe with raw RFM features.
        """
        # If reference date not provided, set to 1 day after the latest transaction
        ref_date = self.reference_date or (df["InvoiceDate"].max() + pd.Timedelta(days=1))

        # Aggregate at CustomerID level
        rfm = df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalSpend", "sum"),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max"),
            UniqueItemsPurchased=("StockCode", "nunique"),
            TotalItemsCount=("Quantity", "sum"),
        ).reset_index()

        # Compute Average Order Value (AOV)
        rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]

        # Customer tenure in days
        rfm["TenureDays"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days

        # Ensure no non-positive monetary values
        rfm["Monetary"] = rfm["Monetary"].clip(lower=0.01)
        rfm["AvgOrderValue"] = rfm["AvgOrderValue"].clip(lower=0.01)
        rfm["Recency"] = rfm["Recency"].clip(lower=1)

        logger.info(f"Computed RFM metrics for {len(rfm):,} unique customers.")
        return rfm

    def transform_and_scale(
        self, rfm_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Applies log1p transformation to reduce positive skewness, followed by
        StandardScaler standardization.

        Parameters
        ----------
        rfm_df : pd.DataFrame
            Customer dataframe with raw RFM columns.

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]
            - Enhanced dataframe with raw + log features
            - Scaled feature matrix (numpy array)
            - Skewness comparison dataframe before and after log transform
        """
        df_transformed = rfm_df.copy()

        skewness_records = []
        for col in self.feature_cols:
            raw_skew = df_transformed[col].skew()
            df_transformed[f"{col}_log"] = np.log1p(df_transformed[col])
            log_skew = df_transformed[f"{col}_log"].skew()

            skewness_records.append({
                "Feature": col,
                "Raw_Skewness": round(raw_skew, 3),
                "Log_Skewness": round(log_skew, 3),
                "Skew_Reduction_Pct": round(abs((raw_skew - log_skew) / raw_skew) * 100, 1),
            })

        skewness_df = pd.DataFrame(skewness_records)

        # Fit and transform scaled features
        scaled_matrix = self.scaler.fit_transform(df_transformed[self.log_feature_cols])

        # Add scaled feature columns to dataframe
        for idx, col in enumerate(self.feature_cols):
            df_transformed[f"{col}_scaled"] = scaled_matrix[:, idx]

        logger.info("RFM feature transformation and StandardScaler normalization complete.")
        return df_transformed, scaled_matrix, skewness_df


def engineer_rfm_features(
    clean_df: pd.DataFrame, reference_date: pd.Timestamp | None = None
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Helper wrapper function to engineer and scale RFM features."""
    engineer = RFMFeatureEngineer(reference_date=reference_date)
    rfm_raw = engineer.compute_rfm(clean_df)
    rfm_transformed, scaled_matrix, skew_df = engineer.transform_and_scale(rfm_raw)
    return rfm_transformed, scaled_matrix, skew_df
