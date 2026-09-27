"""
evaluation.py
-------------
Dimensionality reduction (PCA and t-SNE), silhouette sample analysis,
and projection utilities for cluster evaluation and visualization.
"""

from __future__ import annotations
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """
    Computes dimensionality reduction embeddings (PCA, t-SNE) and
    detailed per-sample silhouette analytics.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pca_2d: PCA | None = None
        self.pca_3d: PCA | None = None

    def compute_pca_2d(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Projects high-dimensional RFM features into 2 principal components.
        """
        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        pca_coords = self.pca_2d.fit_transform(X)

        var_explained = self.pca_2d.explained_variance_ratio_
        total_var = float(np.sum(var_explained))

        meta = {
            "PC1_variance_pct": round(var_explained[0] * 100, 2),
            "PC2_variance_pct": round(var_explained[1] * 100, 2),
            "total_variance_pct": round(total_var * 100, 2),
            "components": self.pca_2d.components_,
        }
        return pca_coords, meta

    def compute_pca_3d(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Projects features into 3 principal components for 3D interactive plots.
        """
        self.pca_3d = PCA(n_components=3, random_state=self.random_state)
        pca_coords = self.pca_3d.fit_transform(X)

        var_explained = self.pca_3d.explained_variance_ratio_
        total_var = float(np.sum(var_explained))

        meta = {
            "PC1_variance_pct": round(var_explained[0] * 100, 2),
            "PC2_variance_pct": round(var_explained[1] * 100, 2),
            "PC3_variance_pct": round(var_explained[2] * 100, 2),
            "total_variance_pct": round(total_var * 100, 2),
        }
        return pca_coords, meta

    def compute_tsne_2d(self, X: np.ndarray, perplexity: float = 35.0) -> np.ndarray:
        """
        Computes 2D t-SNE manifold projection.
        """
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            random_state=self.random_state,
            init="pca",
            learning_rate="auto",
        )
        return tsne.fit_transform(X)

    def calculate_silhouette_samples(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Calculates individual silhouette coefficient per customer.
        """
        return silhouette_samples(X, labels)
