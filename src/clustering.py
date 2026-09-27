"""
clustering.py
-------------
Implements unsupervised clustering algorithms (K-Means, Agglomerative/Hierarchical,
and DBSCAN), optimal cluster search (Elbow & Silhouette analysis), and dendrogram generation.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

logger = logging.getLogger(__name__)


class CustomerClusterer:
    """
    Manages model fitting, hyperparameter evaluation, and cluster assignments
    for multiple unsupervised learning algorithms.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.kmeans_models: Dict[int, KMeans] = {}
        self.optimal_k: int = 5
        self.hierarchical_model: AgglomerativeClustering | None = None
        self.dbscan_model: DBSCAN | None = None

    def evaluate_kmeans_range(
        self, X: np.ndarray, k_range: range = range(2, 11)
    ) -> pd.DataFrame:
        """
        Fits K-Means across a range of k values and calculates Inertia,
        Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.

        Parameters
        ----------
        X : np.ndarray
            Scaled feature matrix.
        k_range : range
            Range of candidate clusters (default: 2 to 10).

        Returns
        -------
        pd.DataFrame
            Evaluation metrics across tested cluster counts.
        """
        metrics = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, init="k-means++", n_init=25, max_iter=400, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            self.kmeans_models[k] = kmeans

            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            inertia = kmeans.inertia_

            metrics.append({
                "k": k,
                "Inertia": inertia,
                "Silhouette_Score": sil,
                "Davies_Bouldin_Index": db,
                "Calinski_Harabasz_Index": ch,
            })

        metrics_df = pd.DataFrame(metrics)
        logger.info("K-Means range evaluation completed successfully.")
        return metrics_df

    def fit_kmeans(self, X: np.ndarray, k: int = 5) -> Tuple[np.ndarray, KMeans]:
        """
        Fits final K-Means model with chosen k.
        """
        self.optimal_k = k
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=25, max_iter=500, random_state=self.random_state)
        labels = kmeans.fit_predict(X)
        self.kmeans_models[k] = kmeans
        return labels, kmeans

    def compute_linkage_matrix(self, X: np.ndarray, method: str = "ward") -> np.ndarray:
        """
        Computes the hierarchical linkage matrix using Ward's minimum variance method.
        """
        # For large datasets (> 3000), sample up to 3000 points to keep dendrogram clean and performant
        if X.shape[0] > 3000:
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(X.shape[0], size=3000, replace=False)
            X_sample = X[sample_indices]
        else:
            X_sample = X

        return linkage(X_sample, method=method)

    def fit_agglomerative(self, X: np.ndarray, n_clusters: int = 5, linkage_method: str = "ward") -> Tuple[np.ndarray, AgglomerativeClustering]:
        """
        Fits Agglomerative Hierarchical Clustering.
        """
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = agg.fit_predict(X)
        self.hierarchical_model = agg
        return labels, agg

    def fit_dbscan(self, X: np.ndarray, eps: float = 0.65, min_samples: int = 15) -> Tuple[np.ndarray, DBSCAN]:
        """
        Fits DBSCAN to discover dense core clusters and identify isolated anomaly/noise points (label -1).
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        self.dbscan_model = dbscan
        return labels, dbscan

    def compare_models(
        self, X: np.ndarray, labels_dict: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compares clustering models using quantitative intrinsic evaluation metrics.
        """
        results = []

        for name, labels in labels_dict.items():
            # For DBSCAN, filter out noise (-1) for fair comparative evaluation
            mask = labels != -1
            n_clusters = len(set(labels[mask]))
            n_noise = np.sum(labels == -1)

            if n_clusters > 1:
                sil = silhouette_score(X[mask], labels[mask])
                db = davies_bouldin_score(X[mask], labels[mask])
                ch = calinski_harabasz_score(X[mask], labels[mask])
            else:
                sil, db, ch = np.nan, np.nan, np.nan

            results.append({
                "Algorithm": name,
                "Number_of_Clusters": n_clusters,
                "Noise_Outlier_Points": int(n_noise),
                "Silhouette_Score": round(sil, 4) if not np.isnan(sil) else None,
                "Davies_Bouldin_Index": round(db, 4) if not np.isnan(db) else None,
                "Calinski_Harabasz_Index": round(ch, 2) if not np.isnan(ch) else None,
            })

        return pd.DataFrame(results)
