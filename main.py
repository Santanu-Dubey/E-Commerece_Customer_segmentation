"""
main.py
-------
End-to-end execution script for the E-Commerce Customer Segmentation pipeline.
Coordinates data preparation, feature engineering, multi-algorithm clustering,
evaluation, business profiling, and artifact exports.
"""

from __future__ import annotations
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("segmentation_pipeline")

from src.data_prep import DataPreparation
from src.feature_engineering import RFMFeatureEngineer
from src.clustering import CustomerClusterer
from src.evaluation import ClusterEvaluator
from src.profiling import SegmentProfiler
from src.visualization import Visualizer


def run_pipeline(
    data_path: str = "data/raw_transactions.csv",
    output_fig_dir: str = "outputs/figures",
    output_tbl_dir: str = "outputs/tables",
    optimal_k: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes the customer segmentation machine learning pipeline.
    """
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(output_tbl_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("=" * 80)
    print(">> STARTING PRODUCTION E-COMMERCE CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Data Preparation & Hygiene
    # -------------------------------------------------------------
    logger.info("PHASE 1: Data Ingestion and Hygiene Cleaning...")
    prep = DataPreparation(raw_data_path=data_path)
    raw_df = prep.load_data()
    clean_df, clean_stats = prep.clean_transactions(raw_df)

    # -------------------------------------------------------------
    # 2. RFM Feature Engineering
    # -------------------------------------------------------------
    logger.info("PHASE 2: RFM Feature Engineering, Log Transform & Normalization...")
    engineer = RFMFeatureEngineer()
    rfm_raw = engineer.compute_rfm(clean_df)
    rfm_df, scaled_matrix, skew_df = engineer.transform_and_scale(rfm_raw)

    skew_path = os.path.join(output_tbl_dir, "rfm_skewness_reduction.csv")
    skew_df.to_csv(skew_path, index=False)
    logger.info(f"Saved feature skewness report: {skew_path}")

    # -------------------------------------------------------------
    # 3. Exploratory Data Analysis & Feature Visualizations
    # -------------------------------------------------------------
    logger.info("PHASE 3: Exploratory Data Analysis and Feature Distributions...")
    viz = Visualizer(output_dir=output_fig_dir)
    viz.plot_raw_distributions(rfm_df)
    viz.plot_transformed_distributions(rfm_df)
    viz.plot_correlation_heatmap(rfm_df)

    # -------------------------------------------------------------
    # 4. Multi-Algorithm Clustering
    # -------------------------------------------------------------
    logger.info("PHASE 4: Multi-Algorithm Clustering Analysis...")
    clusterer = CustomerClusterer(random_state=random_seed)

    # 4a. K-Means evaluation across k=2..10
    logger.info("Evaluating K-Means Inertia & Silhouette curves...")
    kmeans_metrics = clusterer.evaluate_kmeans_range(scaled_matrix, k_range=range(2, 11))
    viz.plot_kmeans_evaluation(kmeans_metrics, optimal_k=optimal_k)

    # 4b. Fit chosen optimal K-Means
    kmeans_labels, kmeans_model = clusterer.fit_kmeans(scaled_matrix, k=optimal_k)
    rfm_df["KMeans_Cluster"] = kmeans_labels

    # 4c. Hierarchical / Agglomerative Clustering
    logger.info("Computing Ward Linkage and Hierarchical Dendrogram...")
    linkage_matrix = clusterer.compute_linkage_matrix(scaled_matrix, method="ward")
    viz.plot_dendrogram(linkage_matrix)
    agg_labels, _ = clusterer.fit_agglomerative(scaled_matrix, n_clusters=optimal_k)
    rfm_df["Agglomerative_Cluster"] = agg_labels

    # 4d. DBSCAN Clustering (Outlier / Anomaly Detection)
    logger.info("Fitting DBSCAN for density clustering and noise identification...")
    dbscan_labels, _ = clusterer.fit_dbscan(scaled_matrix, eps=0.65, min_samples=15)
    rfm_df["DBSCAN_Cluster"] = dbscan_labels

    # -------------------------------------------------------------
    # 5. Model Evaluation & Dimensionality Reduction (PCA / t-SNE)
    # -------------------------------------------------------------
    logger.info("PHASE 5: Dimensionality Reduction & Comparative Model Evaluation...")
    evaluator = ClusterEvaluator(random_state=random_seed)
    pca_2d_coords, pca_meta = evaluator.compute_pca_2d(scaled_matrix)
    rfm_df["PCA1"] = pca_2d_coords[:, 0]
    rfm_df["PCA2"] = pca_2d_coords[:, 1]

    # Model comparison table
    model_labels_dict = {
        f"K-Means (k={optimal_k})": kmeans_labels,
        f"Agglomerative (k={optimal_k})": agg_labels,
        "DBSCAN (Density-Based)": dbscan_labels,
    }
    model_comparison_df = clusterer.compare_models(scaled_matrix, model_labels_dict)
    comp_path = os.path.join(output_tbl_dir, "model_comparison_metrics.csv")
    model_comparison_df.to_csv(comp_path, index=False)
    logger.info(f"Saved model comparison table: {comp_path}")

    # Visual comparisons
    viz.plot_clustering_comparison(pca_2d_coords, model_labels_dict)
    viz.plot_dbscan_analysis(pca_2d_coords, dbscan_labels)

    # -------------------------------------------------------------
    # 6. Business Profiling, Labeling & Marketing Strategy
    # -------------------------------------------------------------
    logger.info("PHASE 6: Segment Profiling, Commercial Metrics & Strategy Mapping...")
    # Base final business segmentation on K-Means
    rfm_df["Cluster"] = rfm_df["KMeans_Cluster"]
    profiler = SegmentProfiler(rfm_df, cluster_col="Cluster")
    rfm_df, cluster_map = profiler.assign_business_labels()

    summary_df = profiler.generate_segment_summary()
    summary_path = os.path.join(output_tbl_dir, "cluster_summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved cluster summary metrics: {summary_path}")

    playbook_df = profiler.generate_business_playbook()
    playbook_path = os.path.join(output_tbl_dir, "segment_business_profiles.csv")
    playbook_df.to_csv(playbook_path, index=False)
    logger.info(f"Saved marketing business playbook: {playbook_path}")

    # -------------------------------------------------------------
    # 7. Final Profile Visualizations & Artifact Exports
    # -------------------------------------------------------------
    logger.info("PHASE 7: Generating Final Profiling Visualizations...")
    viz.plot_rfm_boxplots(rfm_df, segment_col="Segment")
    viz.plot_segment_size_and_revenue(summary_df)
    viz.plot_radar_chart(summary_df)
    viz.plot_interactive_3d_rfm(rfm_df)

    # Export complete enriched customer dataframe
    customer_export_path = "data/customer_rfm_segmented.csv"
    rfm_df.to_csv(customer_export_path, index=False)
    logger.info(f"Saved enriched customer segmentation dataset: {customer_export_path}")

    print("\n" + "=" * 80)
    print(">> SEGMENTATION SUMMARY TABLE")
    print("=" * 80)
    print(
        summary_df[
            [
                "Segment",
                "Customer_Count",
                "Customer_Share_Pct",
                "Revenue_Share_Pct",
                "Recency_Mean",
                "Frequency_Mean",
                "Monetary_Mean",
                "AOV_Mean",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print(">> ALGORITHM COMPARISON")
    print("=" * 80)
    print(model_comparison_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(">> PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print(f"Figures exported to: {output_fig_dir}/")
    print(f"Tables exported to:  {output_tbl_dir}/")
    print(f"Enriched Data:       {customer_export_path}")
    print("=" * 80 + "\n")

    return {
        "rfm_df": rfm_df,
        "summary_df": summary_df,
        "playbook_df": playbook_df,
        "model_comparison_df": model_comparison_df,
        "clean_stats": clean_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="E-Commerce Customer Segmentation ML Pipeline")
    parser.add_argument("--data-path", type=str, default="data/raw_transactions.csv", help="Path to raw transaction dataset")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters for K-Means (default: 5)")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fig-dir", type=str, default="outputs/figures", help="Directory to save figures")
    parser.add_argument("--tbl-dir", type=str, default="outputs/tables", help="Directory to save tables")

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        output_fig_dir=args.fig_dir,
        output_tbl_dir=args.tbl_dir,
        optimal_k=args.k,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
