"""
visualization.py
----------------
Generates publication-quality, aesthetically rich visualizations for customer segmentation:
- Feature distribution plots (raw and log-transformed)
- Correlation heatmaps
- K-Means Elbow & Silhouette evaluation curves
- Hierarchical dendrogram
- Multi-model 2D PCA comparisons
- DBSCAN noise and cluster density
- Segment RFM boxplots & revenue share charts
- Radar / Spider behavioral charts
- Interactive Plotly 3D visualizer
"""

from __future__ import annotations
import os
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

# Premium modern styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

PALETTE_HEX = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77", "#e6ab02", "#a6761d"]


class Visualizer:
    """
    Renders and exports all static and interactive plots for the segmentation pipeline.
    """

    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_raw_distributions(self, df: pd.DataFrame, filename: str = "01_raw_rfm_distributions.png") -> str:
        """Plots histograms and KDEs for raw RFM features to highlight skewness."""
        features = ["Recency", "Frequency", "Monetary", "AvgOrderValue"]
        titles = [
            "Recency (Days since last purchase)",
            "Frequency (Number of orders)",
            "Monetary Value (Total spend $)",
            "Average Order Value ($)",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (col, title) in enumerate(zip(features, titles)):
            skew = df[col].skew()
            sns.histplot(
                df[col],
                kde=True,
                ax=axes[idx],
                color="#1f77b4",
                bins=40,
                alpha=0.6,
                edgecolor="white",
            )
            axes[idx].set_title(f"{title}\n(Skewness: {skew:.2f})", fontweight="bold")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Customer Count")

        plt.suptitle("Raw Feature Distributions (Demonstrating High Right Skewness)", fontsize=16, fontweight="bold", y=0.99)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path

    def plot_transformed_distributions(self, df: pd.DataFrame, filename: str = "02_transformed_rfm_distributions.png") -> str:
        """Plots distributions after log1p transformation."""
        features = ["Recency_log", "Frequency_log", "Monetary_log", "AvgOrderValue_log"]
        titles = [
            "Log1p Recency",
            "Log1p Frequency",
            "Log1p Monetary Value",
            "Log1p Average Order Value",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (col, title) in enumerate(zip(features, titles)):
            skew = df[col].skew()
            sns.histplot(
                df[col],
                kde=True,
                ax=axes[idx],
                color="#2ca02c",
                bins=40,
                alpha=0.6,
                edgecolor="white",
            )
            axes[idx].set_title(f"{title}\n(Skewness: {skew:.2f})", fontweight="bold")
            axes[idx].set_xlabel(f"{col}")
            axes[idx].set_ylabel("Customer Count")

        plt.suptitle("Log1p-Transformed Feature Distributions (Normalized Bell Curves)", fontsize=16, fontweight="bold", y=0.99)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path

    def plot_correlation_heatmap(self, df: pd.DataFrame, filename: str = "03_rfm_correlation_heatmap.png") -> str:
        """Plots Pearson correlation heatmap between RFM features."""
        cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "Recency_log", "Frequency_log", "Monetary_log", "AvgOrderValue_log"]
        corr = df[cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=1.0,
            cbar_kws={"shrink": 0.8},
            annot=True,
            fmt=".2f",
            ax=ax,
        )
        plt.title("Correlation Heatmap: Raw vs. Log-Transformed RFM Features", fontsize=15, fontweight="bold", pad=15)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path

    def plot_kmeans_evaluation(self, metrics_df: pd.DataFrame, optimal_k: int = 5, filename: str = "04_kmeans_elbow_silhouette.png") -> str:
        """Plots Elbow method (Inertia) and Silhouette Scores across k."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Inertia / Elbow Curve
        ax1.plot(metrics_df["k"], metrics_df["Inertia"], marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
        ax1.axvline(x=optimal_k, color="#d62728", linestyle="--", linewidth=1.8, label=f"Selected Optimal k = {optimal_k}")
        ax1.set_title("Elbow Method: Inertia (SSE) vs. Number of Clusters (k)", fontweight="bold")
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
        ax1.set_xticks(metrics_df["k"])
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        # Silhouette Score Curve
        ax2.plot(metrics_df["k"], metrics_df["Silhouette_Score"], marker="s", color="#2ca02c", linewidth=2.5, markersize=8)
        ax2.axvline(x=optimal_k, color="#d62728", linestyle="--", linewidth=1.8, label=f"Selected Optimal k = {optimal_k}")
        ax2.set_title("Silhouette Score vs. Number of Clusters (k)", fontweight="bold")
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Mean Silhouette Coefficient")
        ax2.set_xticks(metrics_df["k"])
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        plt.suptitle("K-Means Hyperparameter Optimization Diagnostics", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path

    def plot_dendrogram(self, linkage_matrix: np.ndarray, color_threshold: float = 35.0, filename: str = "05_hierarchical_dendrogram.png") -> str:
        """Plots hierarchical clustering dendrogram."""
        fig, ax = plt.subplots(figsize=(14, 7))
        dendrogram(
            linkage_matrix,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90.0,
            leaf_font_size=10.0,
            show_contracted=True,
            color_threshold=color_threshold,
            ax=ax,
        )
        ax.set_title("Hierarchical Clustering Dendrogram (Ward's Minimum Variance Linkage)", fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("Cluster Sample Index (Truncated to Top 30 Merge Nodes)")
        ax.set_ylabel("Ward Euclidean Linkage Distance")
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path

    def plot_clustering_comparison(
        self,
        pca_coords: np.ndarray,
        labels_dict: Dict[str, np.ndarray],
        segment_names: List[str] | None = None,
        filename: str = "06_clustering_comparison_pca.png",
    ) -> str:
        """Plots 2x2 comparison grid of PCA projections across clustering algorithms."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()

        for idx, (algo_name, labels) in enumerate(labels_dict.items()):
            ax = axes[idx]
            unique_labels = sorted(list(set(labels)))

            for label in unique_labels:
                mask = labels == label
                if label == -1:
                    ax.scatter(
                        pca_coords[mask, 0],
                        pca_coords[mask, 1],
                        c="#999999",
                        label="Noise / Outliers",
                        alpha=0.4,
                        s=20,
                        marker="x",
                    )
                else:
                    lbl_text = segment_names[label] if (segment_names and label < len(segment_names)) else f"Cluster {label}"
                    ax.scatter(
                        pca_coords[mask, 0],
                        pca_coords[mask, 1],
                        label=lbl_text,
                        alpha=0.75,
                        s=35,
                        edgecolors="none",
                    )

            ax.set_title(f"{algo_name} (2D PCA Projection)", fontweight="bold")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.legend(loc="upper right", frameon=True, fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

        plt.suptitle("Comparative Evaluation of Clustering Approaches on 2D PCA Projection", fontsize=16, fontweight="bold", y=0.99)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path

    def plot_dbscan_analysis(
        self,
        pca_coords: np.ndarray,
        dbscan_labels: np.ndarray,
        filename: str = "07_dbscan_noise_and_density.png",
    ) -> str:
        """Visualizes DBSCAN density clusters and isolated noise points."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: PCA view
        noise_mask = dbscan_labels == -1
        core_mask = ~noise_mask

        ax1.scatter(pca_coords[core_mask, 0], pca_coords[core_mask, 1], c=dbscan_labels[core_mask], cmap="tab10", s=35, alpha=0.7, label="Core Clusters")
        ax1.scatter(pca_coords[noise_mask, 0], pca_coords[noise_mask, 1], c="crimson", marker="x", s=40, alpha=0.85, label=f"Flagged Anomalies/Noise (n={np.sum(noise_mask)})")
        ax1.set_title("DBSCAN Cluster Separation & Outlier Detection", fontweight="bold")
        ax1.set_xlabel("Principal Component 1")
        ax1.set_ylabel("Principal Component 2")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Subplot 2: Cluster size breakdown
        counts = pd.Series(dbscan_labels).value_counts().sort_index()
        labels_str = [f"Cluster {idx}" if idx != -1 else "Noise (-1)" for idx in counts.index]
        colors = ["crimson" if idx == -1 else "#1f77b4" for idx in counts.index]

        sns.barplot(x=labels_str, y=counts.values, palette=colors, ax=ax2, hue=labels_str, legend=False)
        ax2.set_title("DBSCAN Membership Count per Cluster", fontweight="bold")
        ax2.set_xlabel("Cluster / Anomaly ID")
        ax2.set_ylabel("Number of Customers")
        for p in ax2.patches:
            ax2.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path

    def plot_rfm_boxplots(self, df: pd.DataFrame, segment_col: str = "Segment", filename: str = "08_final_cluster_rfm_boxplots.png") -> str:
        """Generates log-scaled boxplots for each RFM dimension across business segments."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        metrics = [
            ("Recency", "Recency (Days - Log Scale)", True),
            ("Frequency", "Frequency (Order Count - Log Scale)", True),
            ("Monetary", "Monetary Value ($ Total Spend - Log Scale)", True),
            ("AvgOrderValue", "Average Order Value ($ - Log Scale)", True),
        ]

        segments_order = df.groupby(segment_col)["Monetary"].median().sort_values(ascending=False).index.tolist()

        for idx, (col, title, log_scale) in enumerate(metrics):
            ax = axes[idx]
            sns.boxplot(
                data=df,
                x=segment_col,
                y=col,
                order=segments_order,
                palette="Set2",
                ax=ax,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"},
            )
            if log_scale:
                ax.set_yscale("log")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(col)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, linestyle=":", alpha=0.5)

        plt.suptitle("Customer RFM Distribution Profiles by Final Business Segment", fontsize=16, fontweight="bold", y=0.99)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path

    def plot_segment_size_and_revenue(self, summary_df: pd.DataFrame, filename: str = "09_segment_size_distribution.png") -> str:
        """Visualizes customer share vs revenue contribution per segment."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        df_sorted = summary_df.sort_values("Revenue_Share_Pct", ascending=True)

        # Plot Customer Share
        sns.barplot(
            data=df_sorted,
            y="Segment",
            x="Customer_Share_Pct",
            palette="Blues_d",
            ax=ax1,
            hue="Segment",
            legend=False,
        )
        ax1.set_title("Customer Volume Share (%)", fontweight="bold", fontsize=14)
        ax1.set_xlabel("Share of Total Customers (%)")
        ax1.set_ylabel("")
        for p in ax1.patches:
            val = p.get_width()
            ax1.annotate(f"{val:.1f}%", (val + 0.5, p.get_y() + p.get_height() / 2.0),
                         ha="left", va="center", fontsize=10, fontweight="bold")

        # Plot Revenue Share
        sns.barplot(
            data=df_sorted,
            y="Segment",
            x="Revenue_Share_Pct",
            palette="Greens_d",
            ax=ax2,
            hue="Segment",
            legend=False,
        )
        ax2.set_title("Revenue Contribution Share (%)", fontweight="bold", fontsize=14)
        ax2.set_xlabel("Share of Total Revenue (%)")
        ax2.set_ylabel("")
        for p in ax2.patches:
            val = p.get_width()
            ax2.annotate(f"{val:.1f}%", (val + 0.5, p.get_y() + p.get_height() / 2.0),
                         ha="left", va="center", fontsize=10, fontweight="bold")

        plt.suptitle("Customer Volume vs. Revenue Contribution by Segment (Pareto Dynamics)", fontsize=16, fontweight="bold", y=0.99)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path

    def plot_radar_chart(self, summary_df: pd.DataFrame, filename: str = "10_radar_chart_profiles.png") -> str:
        """Plots multi-axis radar chart showing normalized segment behavioral characteristics."""
        categories = ["Recency (Inverted)", "Frequency", "Monetary", "AvgOrderValue"]
        N = len(categories)

        # Compute normalized scores 0-1 across segments
        radar_data = summary_df.copy()
        # Invert recency so that higher = more recent (better)
        max_rec = radar_data["Recency_Mean"].max()
        radar_data["Norm_Recency"] = (max_rec - radar_data["Recency_Mean"]) / (max_rec - radar_data["Recency_Mean"].min() + 1e-5)
        radar_data["Norm_Frequency"] = (radar_data["Frequency_Mean"] - radar_data["Frequency_Mean"].min()) / (radar_data["Frequency_Mean"].max() - radar_data["Frequency_Mean"].min() + 1e-5)
        radar_data["Norm_Monetary"] = (radar_data["Monetary_Mean"] - radar_data["Monetary_Mean"].min()) / (radar_data["Monetary_Mean"].max() - radar_data["Monetary_Mean"].min() + 1e-5)
        radar_data["Norm_AOV"] = (radar_data["AOV_Mean"] - radar_data["AOV_Mean"].min()) / (radar_data["AOV_Mean"].max() - radar_data["AOV_Mean"].min() + 1e-5)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, size=11, fontweight="bold")
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.50, 0.75, 1.0], ["25%", "50%", "75%", "100%"], color="grey", size=9)
        plt.ylim(0, 1.1)

        colors = sns.color_palette("tab10", len(radar_data))

        for idx, row in radar_data.iterrows():
            values = [row["Norm_Recency"], row["Norm_Frequency"], row["Norm_Monetary"], row["Norm_AOV"]]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle="solid", label=row["Segment"], color=colors[idx])
            ax.fill(angles, values, color=colors[idx], alpha=0.15)

        plt.title("Normalized Behavioral Radar Profile by Customer Segment", size=15, fontweight="bold", y=1.08)
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return out_path

    def plot_interactive_3d_rfm(self, df: pd.DataFrame, filename: str = "11_interactive_3d_rfm.html") -> str | None:
        """Generates interactive 3D HTML scatter plot using Plotly."""
        if not HAS_PLOTLY:
            logger.warning("Plotly not installed. Skipping interactive 3D plot.")
            return None

        sample_df = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df

        fig = px.scatter_3d(
            sample_df,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color="Segment",
            log_x=True,
            log_y=True,
            log_z=True,
            hover_name="Segment",
            hover_data={
                "CustomerID": True,
                "Recency": ":.0f days",
                "Frequency": ":.0f orders",
                "Monetary": ":$.2f",
                "AvgOrderValue": ":$.2f",
            },
            title="Interactive 3D Customer Segmentation Space (Log Scale: Recency vs Frequency vs Monetary)",
            opacity=0.8,
            template="plotly_white",
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            scene=dict(
                xaxis_title="Recency (Days - Log)",
                yaxis_title="Frequency (Orders - Log)",
                zaxis_title="Monetary Spend ($ - Log)",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        out_path = os.path.join(self.output_dir, filename)
        fig.write_html(out_path)
        logger.info(f"Saved interactive HTML: {out_path}")
        return out_path
