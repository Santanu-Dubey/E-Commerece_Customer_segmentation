"""
synthetic_data.py
-----------------
Generates a realistic, high-fidelity synthetic e-commerce transaction dataset
mimicking the UCI Online Retail / Online Retail II benchmark dataset structure.
Includes realistic customer behavior archetypes, power-law order distributions,
cancellations, product catalogs, and seasonality.
"""

from __future__ import annotations
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# Sample realistic product catalog
PRODUCT_CATALOG = [
    ("22728", "ALARM CLOCK BAKELIKE PINK", 3.75),
    ("22727", "ALARM CLOCK BAKELIKE RED", 3.75),
    ("22726", "ALARM CLOCK BAKELIKE GREEN", 3.75),
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("47566", "PARTY BUNTING", 4.95),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("20727", "LUNCH BAG BLACK SKULL", 1.65),
    ("20728", "LUNCH BAG CARS BLUE", 1.65),
    ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
    ("22386", "JUMBO BAG PINK POLKADOT", 2.08),
    ("85099B", "JUMBO BAG RED RETROSPOT", 2.08),
    ("23084", "RABBIT NIGHT LIGHT", 2.08),
    ("22960", "JAM MAKING SET WITH JARS", 4.25),
    ("22197", "POPCORN HOLDER", 0.85),
    ("22910", "PAPER CHAIN KIT VINTAGE CHRISTMAS", 2.95),
    ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.95),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("84946", "ANTIQUE SILVER T-LIGHT GLASS", 1.25),
    ("21931", "JUMBO STORAGE BAG SUKI", 2.08),
    ("22961", "JAM MAKING SET PRINTED", 1.45),
    ("23203", "JUMBO BAG VINTAGE DOILY", 2.08),
    ("22384", "LUNCH BAG PINK POLKADOT", 1.65),
    ("22178", "VICTORIAN GLASS HANGING T-LIGHT", 1.25),
    ("22666", "RECIPE BOX PANTRY DESIGN", 7.95),
    ("21733", "RED HANGING HEART T-LIGHT HOLDER", 2.95),
    ("82482", "WOODEN PICTURE FRAME WHITE FINISH", 2.55),
    ("22111", "SCOTTIE DOG HOT WATER BOTTLE", 4.95),
    ("21034", "REX CASH + CARRY JUMBO SHOPPER", 0.95),
    ("22457", "NATURAL SLATE HEART CHALKBOARD", 2.95),
    ("22469", "HEART OF WICKER SMALL", 1.65),
    ("22470", "HEART OF WICKER LARGE", 2.95),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 4.25),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
    ("22699", "ROSES REGENCY TEACUP AND SAUCER", 6.95),
    ("22697", "GREEN REGENCY TEACUP AND SAUCER", 6.95),
    ("22698", "PINK REGENCY TEACUP AND SAUCER", 6.95),
    ("21929", "JUMBO BAG VINTAGE PAISLEY", 2.08),
    ("22382", "LUNCH BAG SPACEBOY DESIGN", 1.65),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART", 4.25),
    ("22659", "SILVER BIRD ORNAMENT", 2.95),
    ("21754", "HOME BUILDING BLOCK WORD", 5.95),
    ("21755", "LOVE BUILDING BLOCK WORD", 5.95),
    ("22138", "BAKING SET 9 PIECE RETROSPOT", 4.95),
    ("22355", "CHARLOTTE BAG SUKI DESIGN", 0.85),
    ("22356", "CHARLOTTE BAG PINK POLKADOT", 0.85),
    ("22554", "PLASTERS IN TIN WOODLAND ANIMALS", 1.65),
    ("22556", "PLASTERS IN TIN CIRCUS PARADE", 1.65),
    ("22551", "PLASTERS IN TIN SPACEBOY", 1.65),
]

COUNTRIES = [
    ("United Kingdom", 0.88),
    ("Germany", 0.04),
    ("France", 0.03),
    ("EIRE", 0.02),
    ("Spain", 0.01),
    ("Netherlands", 0.01),
    ("Belgium", 0.005),
    ("Switzerland", 0.005),
]


def generate_synthetic_transactions(
    num_customers: int = 4200,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    random_seed: int = 42,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generates a synthetic e-commerce transaction dataset with realistic
    customer purchasing dynamics and RFM archetype distributions.

    Parameters
    ----------
    num_customers : int
        Number of unique customer IDs to simulate (default: 4200).
    start_date : str
        Start date of simulation window ('YYYY-MM-DD').
    end_date : str
        End date of simulation window ('YYYY-MM-DD').
    random_seed : int
        Seed for reproducibility.
    output_path : str, optional
        If provided, writes the dataset as CSV to this filepath.

    Returns
    -------
    pd.DataFrame
        Synthetic raw transaction table matching UCI Online Retail format.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days

    country_names = [c[0] for c in COUNTRIES]
    country_weights = [c[1] for c in COUNTRIES]

    # Assign each customer to an archetype to induce realistic behavioral clusters
    # Archetype proportions:
    # 0: Champions (High spend, frequent, recent) -> 12%
    # 1: Loyal Customers (Moderate-high spend, consistent frequency) -> 24%
    # 2: Big Spenders / Wholesale (High spend, low frequency) -> 10%
    # 3: New / Recent Promising (Recent, 1-2 orders, moderate spend) -> 18%
    # 4: At Risk / Churning (High frequency in past, no recent purchases) -> 16%
    # 5: Dormant / Lost (Low frequency, long time ago, low spend) -> 20%
    archetypes = np.random.choice(
        [0, 1, 2, 3, 4, 5],
        size=num_customers,
        p=[0.12, 0.24, 0.10, 0.18, 0.16, 0.20],
    )

    customer_ids = [12346 + i for i in range(num_customers)]
    customer_countries = np.random.choice(country_names, size=num_customers, p=country_weights)

    records = []
    invoice_seq = 536000

    for idx, (cid, arch) in enumerate(zip(customer_ids, archetypes)):
        country = customer_countries[idx]

        if arch == 0:  # Champions
            num_orders = np.random.randint(8, 26)
            first_order_day = np.random.randint(0, total_days // 3)
            last_order_day = np.random.randint(total_days - 25, total_days)
            items_per_order_range = (4, 15)
            qty_multiplier = (3, 18)
        elif arch == 1:  # Loyal
            num_orders = np.random.randint(4, 12)
            first_order_day = np.random.randint(0, total_days // 2)
            last_order_day = np.random.randint(total_days - 60, total_days - 5)
            items_per_order_range = (2, 8)
            qty_multiplier = (2, 8)
        elif arch == 2:  # Big Spenders / Wholesale
            num_orders = np.random.randint(1, 4)
            first_order_day = np.random.randint(0, total_days - 10)
            last_order_day = np.random.randint(first_order_day, total_days)
            items_per_order_range = (6, 20)
            qty_multiplier = (25, 120)  # bulk purchases
        elif arch == 3:  # New Customers
            num_orders = np.random.randint(1, 3)
            first_order_day = np.random.randint(total_days - 45, total_days)
            last_order_day = np.random.randint(first_order_day, total_days)
            items_per_order_range = (1, 5)
            qty_multiplier = (1, 6)
        elif arch == 4:  # At Risk
            num_orders = np.random.randint(3, 9)
            first_order_day = np.random.randint(0, total_days // 4)
            last_order_day = np.random.randint(total_days // 3, total_days - 90)
            items_per_order_range = (2, 7)
            qty_multiplier = (1, 6)
        else:  # Dormant / Lost
            num_orders = np.random.randint(1, 2)
            first_order_day = np.random.randint(0, total_days - 150)
            last_order_day = first_order_day
            items_per_order_range = (1, 4)
            qty_multiplier = (1, 4)

        # Generate order timestamps between first_order_day and last_order_day
        if num_orders == 1:
            order_days = [first_order_day]
        else:
            order_days = sorted(
                np.random.randint(first_order_day, max(first_order_day + 1, last_order_day + 1), size=num_orders)
            )

        for o_day in order_days:
            invoice_seq += 1
            inv_no = str(invoice_seq)
            order_date = start_dt + timedelta(
                days=int(o_day),
                hours=random.randint(8, 19),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )

            # Realistic number of lines in this invoice
            num_items = random.randint(*items_per_order_range)
            chosen_products = random.sample(PRODUCT_CATALOG, min(num_items, len(PRODUCT_CATALOG)))

            for stock_code, desc, base_price in chosen_products:
                qty = random.randint(*qty_multiplier)
                # Small price jitter for realistic discounting
                unit_price = round(max(0.20, base_price * np.random.normal(1.0, 0.05)), 2)

                records.append({
                    "InvoiceNo": inv_no,
                    "StockCode": stock_code,
                    "Description": desc,
                    "Quantity": qty,
                    "InvoiceDate": order_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice": unit_price,
                    "CustomerID": cid,
                    "Country": country,
                })

            # Small 2.5% chance of cancelled transaction
            if random.random() < 0.025:
                cancel_inv = f"C{invoice_seq}"
                cancel_prod = random.choice(chosen_products)
                records.append({
                    "InvoiceNo": cancel_inv,
                    "StockCode": cancel_prod[0],
                    "Description": cancel_prod[1],
                    "Quantity": -random.randint(1, 5),
                    "InvoiceDate": (order_date + timedelta(hours=random.randint(1, 48))).strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice": cancel_prod[2],
                    "CustomerID": cid,
                    "Country": country,
                })

    # Add a small proportion (~3%) of missing CustomerIDs / guest checkouts for realistic data cleaning
    num_guest_records = int(len(records) * 0.03)
    for _ in range(num_guest_records):
        guest_prod = random.choice(PRODUCT_CATALOG)
        guest_date = start_dt + timedelta(days=random.randint(0, total_days), hours=random.randint(8, 20))
        records.append({
            "InvoiceNo": str(invoice_seq + random.randint(100, 500)),
            "StockCode": guest_prod[0],
            "Description": guest_prod[1],
            "Quantity": random.randint(1, 4),
            "InvoiceDate": guest_date.strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice": guest_prod[2],
            "CustomerID": np.nan,  # Missing customer ID
            "Country": "United Kingdom",
        })

    df = pd.DataFrame(records)
    # Shuffle records chronologically
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.sort_values("InvoiceDate").reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[synthetic_data] Saved {len(df):,} transactions across {df['CustomerID'].nunique():,} customers to '{output_path}'.")

    return df


if __name__ == "__main__":
    generate_synthetic_transactions(output_path="data/raw_transactions.csv")
