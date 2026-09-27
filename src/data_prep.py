"""
data_prep.py
------------
Handles data ingestion, validation, and cleaning for e-commerce transactional data.
Removes invalid records, cancellations, guest transactions, and duplicates,
and calculates line-item total monetary values.
"""

from __future__ import annotations
import os
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from src.synthetic_data import generate_synthetic_transactions

logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Cleans and prepares transactional data for RFM customer segmentation.
    """

    def __init__(self, raw_data_path: str = "data/raw_transactions.csv"):
        self.raw_data_path = raw_data_path
        self.cleaning_stats: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """
        Loads the transaction dataset. If the file does not exist,
        automatically generates a realistic synthetic dataset.
        """
        if not os.path.exists(self.raw_data_path):
            logger.info(f"File not found at '{self.raw_data_path}'. Generating synthetic dataset...")
            df = generate_synthetic_transactions(output_path=self.raw_data_path)
        else:
            logger.info(f"Loading transaction dataset from '{self.raw_data_path}'...")
            df = pd.read_csv(self.raw_data_path)

        return df

    def clean_transactions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans the transaction dataframe by executing systematic data hygiene:
        1. Parse datetimes
        2. Remove missing CustomerID records
        3. Remove cancelled invoices (InvoiceNo starting with 'C')
        4. Remove non-positive quantities and unit prices
        5. Remove exact duplicate records
        6. Compute line-item TotalSpend (Quantity * UnitPrice)

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction dataframe.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Cleaned dataframe and a dictionary of data cleaning statistics.
        """
        raw_count = len(df)
        raw_customers = df["CustomerID"].nunique(dropna=True)

        # 1. Parse dates
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        # 2. Filter missing CustomerID
        missing_customers = df["CustomerID"].isna().sum()
        df_clean = df.dropna(subset=["CustomerID"]).copy()
        df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)

        # 3. Filter cancellations
        df_clean["InvoiceNo"] = df_clean["InvoiceNo"].astype(str)
        cancelled_mask = df_clean["InvoiceNo"].str.startswith("C") | (df_clean["Quantity"] <= 0)
        num_cancelled = cancelled_mask.sum()
        df_clean = df_clean[~cancelled_mask].copy()

        # 4. Filter invalid prices
        invalid_price_mask = df_clean["UnitPrice"] <= 0
        num_invalid_price = invalid_price_mask.sum()
        df_clean = df_clean[~invalid_price_mask].copy()

        # 5. Remove duplicates
        num_duplicates = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

        # 6. Feature creation
        df_clean["TotalSpend"] = df_clean["Quantity"] * df_clean["UnitPrice"]

        final_count = len(df_clean)
        final_customers = df_clean["CustomerID"].nunique()

        self.cleaning_stats = {
            "initial_rows": raw_count,
            "missing_customer_ids_removed": int(missing_customers),
            "cancellations_negative_qty_removed": int(num_cancelled),
            "invalid_unit_prices_removed": int(num_invalid_price),
            "duplicates_removed": int(num_duplicates),
            "final_clean_rows": final_count,
            "retention_rate_pct": round((final_count / raw_count) * 100, 2),
            "unique_customers": final_customers,
            "min_date": df_clean["InvoiceDate"].min().strftime("%Y-%m-%d"),
            "max_date": df_clean["InvoiceDate"].max().strftime("%Y-%m-%d"),
        }

        logger.info(f"Data Cleaning Completed: {final_count:,} valid records retained ({self.cleaning_stats['retention_rate_pct']}%) across {final_customers:,} unique customers.")
        return df_clean, self.cleaning_stats


def load_and_clean_data(file_path: str = "data/raw_transactions.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Helper wrapper function to load and clean data in one call."""
    prep = DataPreparation(raw_data_path=file_path)
    raw_df = prep.load_data()
    clean_df, stats = prep.clean_transactions(raw_df)
    return clean_df, stats
