"""
profiling.py
------------
Analyzes cluster centroids and aggregate statistics to assign clear business labels,
compute commercial metrics (revenue share, customer share), and map tailored
marketing strategies and KPI recommendations for each customer segment.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Marketing Playbook definitions mapped to archetypes
MARKETING_PLAYBOOK = {
    "Champions": {
        "Description": "High-value, highly active customers with high spend, frequent orders, and very recent purchases. They represent the core revenue engine and brand advocates.",
        "Marketing_Action": "Reward with exclusive VIP perks, early access to new product drops, personalized thank-you notes, and referral incentives. Avoid aggressive discounts.",
        "Preferred_Channel": "Direct VIP concierge email, dedicated SMS alerts, exclusive loyalty portal.",
        "Key_KPIs": "Customer Lifetime Value (CLV), Net Promoter Score (NPS), Referral Rate, Share of Wallet.",
        "Priority": "P1 - Critical Retention & Advocacy",
    },
    "Loyal / Regulars": {
        "Description": "Consistent, trustworthy repeat buyers with steady purchase cadence and moderate-to-high basket sizes. High brand affinity.",
        "Marketing_Action": "Enroll in tiered loyalty/rewards programs. Upsell complementary product bundles and cross-sell trending categories with personalized recommendations.",
        "Preferred_Channel": "Weekly curated email digests, in-app push notifications, targeted social retargeting.",
        "Key_KPIs": "Repeat Purchase Rate, Average Order Value (AOV), Category Cross-Purchase Rate.",
        "Priority": "P2 - Value Expansion & Cross-Sell",
    },
    "Big Spenders / Wholesale": {
        "Description": "High total monetary spend and large basket values, but lower order frequency. Likely bulk purchasers, gift buyers, or business clients.",
        "Marketing_Action": "Offer volume-tier wholesale pricing, dedicated account management, bulk reorder subscriptions, and seasonal pre-order catalogs.",
        "Preferred_Channel": "B2B sales outreach, tailored email catalogs, invoiced payment terms.",
        "Key_KPIs": "Average Order Value (AOV), Units per Transaction (UPT), Reorder Interval.",
        "Priority": "P2 - Order Size & B2B Expansion",
    },
    "New & Promising": {
        "Description": "Recently acquired customers with 1-2 purchases and positive basket sizes. In the critical onboarding window.",
        "Marketing_Action": "Deploy automated welcome onboarding email drip series, provide second-purchase incentives with expiration urgency (e.g. 15% off next 14 days), and educational product guides.",
        "Preferred_Channel": "Onboarding email journey, browser push, welcome SMS.",
        "Key_KPIs": "First-to-Second Purchase Conversion Rate, Time-to-Second Order, Unsubscribe Rate.",
        "Priority": "P1 - Immediate Activation",
    },
    "At Risk / Churning": {
        "Description": "Previously frequent and valuable shoppers who have not made a purchase in an extended period. High likelihood of attrition to competitors.",
        "Marketing_Action": "Trigger aggressive win-back email sequence with high-value time-sensitive discounts ('We Miss You - $20 Off'), customer satisfaction surveys, and re-engagement campaigns.",
        "Preferred_Channel": "Dynamic win-back email drip, remarketing display ads, SMS reactivation.",
        "Key_KPIs": "Reactivation Rate, Churn Rate, Win-Back Cost per Acquisition (CPA).",
        "Priority": "P1 - Churn Mitigation",
    },
    "Dormant / Lost": {
        "Description": "Infrequent buyers with very high recency (long inactivity) and low overall spend. Lowest response probability.",
        "Marketing_Action": "Low-cost programmatic re-engagement or seasonal clearance campaigns. Suppress from expensive paid channels to protect ROI and sender reputation.",
        "Preferred_Channel": "Automated bi-monthly clearance newsletters, low-cost programmatic retargeting.",
        "Key_KPIs": "Reactivation ROI, List Hygiene / Bounce Rate, Unsubscribe Rate.",
        "Priority": "P3 - Low-Cost Reactivation / List Pruning",
    },
}


class SegmentProfiler:
    """
    Profiles clustered customer groups, computes summary metrics,
    maps business semantics, and generates commercial reports.
    """

    def __init__(self, rfm_df: pd.DataFrame, cluster_col: str = "Cluster"):
        self.rfm_df = rfm_df.copy()
        self.cluster_col = cluster_col

    def assign_business_labels(self) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """
        Assigns intuitive, business-friendly segment names to each cluster ID
        based on cluster centroid characteristics in (Recency, Frequency, Monetary).
        """
        cluster_stats = self.rfm_df.groupby(self.cluster_col).agg(
            Recency_mean=("Recency", "mean"),
            Frequency_mean=("Frequency", "mean"),
            Monetary_mean=("Monetary", "mean"),
            AOV_mean=("AvgOrderValue", "mean"),
        ).reset_index()

        cluster_map: Dict[int, str] = {}
        assigned_labels = set()

        # Rank clusters by characteristics
        # Rank by recency ascending (lowest recency = most recent = best)
        # Rank by monetary and frequency descending (highest = best)
        stats = cluster_stats.copy()
        n_clusters = len(stats)

        # Let's map dynamically according to cluster properties
        for _, row in stats.iterrows():
            c_id = int(row[self.cluster_col])
            r = row["Recency_mean"]
            f = row["Frequency_mean"]
            m = row["Monetary_mean"]
            aov = row["AOV_mean"]

            # Compute relative percentiles across clusters
            r_rank = (stats["Recency_mean"] > r).sum() / (n_clusters - 1) if n_clusters > 1 else 0.5 # 1.0 = lowest recency
            f_rank = (stats["Frequency_mean"] < f).sum() / (n_clusters - 1) if n_clusters > 1 else 0.5 # 1.0 = highest freq
            m_rank = (stats["Monetary_mean"] < m).sum() / (n_clusters - 1) if n_clusters > 1 else 0.5 # 1.0 = highest monetary
            aov_rank = (stats["AOV_mean"] < aov).sum() / (n_clusters - 1) if n_clusters > 1 else 0.5

            if r_rank >= 0.75 and (f_rank >= 0.7 or m_rank >= 0.7) and "Champions" not in assigned_labels:
                label = "Champions"
            elif aov_rank >= 0.8 and f_rank < 0.6 and "Big Spenders / Wholesale" not in assigned_labels:
                label = "Big Spenders / Wholesale"
            elif r_rank >= 0.6 and f_rank >= 0.4 and "Loyal / Regulars" not in assigned_labels:
                label = "Loyal / Regulars"
            elif r_rank >= 0.6 and f_rank < 0.4 and "New & Promising" not in assigned_labels:
                label = "New & Promising"
            elif r_rank < 0.4 and f_rank >= 0.3 and "At Risk / Churning" not in assigned_labels:
                label = "At Risk / Churning"
            elif "Dormant / Lost" not in assigned_labels:
                label = "Dormant / Lost"
            else:
                # Fallback to standard archetype list
                available = [k for k in MARKETING_PLAYBOOK.keys() if k not in assigned_labels]
                label = available[0] if available else f"Segment {c_id}"

            assigned_labels.add(label)
            cluster_map[c_id] = label

        # Map labels into dataframe
        self.rfm_df["Segment"] = self.rfm_df[self.cluster_col].map(cluster_map)
        return self.rfm_df, cluster_map

    def generate_segment_summary(self) -> pd.DataFrame:
        """
        Generates a summary table detailing customer counts, revenue share,
        and statistical measures (mean, median) for each segment.
        """
        total_customers = len(self.rfm_df)
        total_revenue = self.rfm_df["Monetary"].sum()

        summary = self.rfm_df.groupby(["Segment", self.cluster_col]).agg(
            Customer_Count=("CustomerID", "count"),
            Total_Revenue=("Monetary", "sum"),
            Recency_Mean=("Recency", "mean"),
            Recency_Median=("Recency", "median"),
            Frequency_Mean=("Frequency", "mean"),
            Frequency_Median=("Frequency", "median"),
            Monetary_Mean=("Monetary", "mean"),
            Monetary_Median=("Monetary", "median"),
            AOV_Mean=("AvgOrderValue", "mean"),
            AOV_Median=("AvgOrderValue", "median"),
        ).reset_index()

        summary["Customer_Share_Pct"] = round((summary["Customer_Count"] / total_customers) * 100, 2)
        summary["Revenue_Share_Pct"] = round((summary["Total_Revenue"] / total_revenue) * 100, 2)

        # Reorder and format
        summary = summary.sort_values("Total_Revenue", ascending=False).reset_index(drop=True)
        return summary

    def generate_business_playbook(self) -> pd.DataFrame:
        """
        Generates the full strategic marketing playbook matrix combining
        cluster statistics with actionable marketing recommendations.
        """
        summary_df = self.generate_segment_summary()

        playbook_rows = []
        for _, row in summary_df.iterrows():
            seg = row["Segment"]
            playbook_info = MARKETING_PLAYBOOK.get(seg, {
                "Description": "General behavioral segment.",
                "Marketing_Action": "Standard marketing campaigns.",
                "Preferred_Channel": "Email & Web",
                "Key_KPIs": "Conversion Rate, Engagement",
                "Priority": "P2 - Standard",
            })

            playbook_rows.append({
                "Cluster_ID": row[self.cluster_col],
                "Segment_Name": seg,
                "Customer_Count": int(row["Customer_Count"]),
                "Customer_Share_Pct": f"{row['Customer_Share_Pct']}%",
                "Revenue_Share_Pct": f"{row['Revenue_Share_Pct']}%",
                "Avg_Recency_Days": f"{row['Recency_Mean']:.1f}",
                "Avg_Frequency_Orders": f"{row['Frequency_Mean']:.1f}",
                "Avg_Monetary_Spend": f"${row['Monetary_Mean']:,.2f}",
                "Avg_Order_Value": f"${row['AOV_Mean']:,.2f}",
                "Priority": playbook_info["Priority"],
                "Business_Description": playbook_info["Description"],
                "Recommended_Action": playbook_info["Marketing_Action"],
                "Preferred_Channel": playbook_info["Preferred_Channel"],
                "Target_KPIs": playbook_info["Key_KPIs"],
            })

        return pd.DataFrame(playbook_rows)
