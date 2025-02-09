import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import seaborn as sns
import os
import json
from pathlib import Path
import requests
from typing import Tuple, List, Dict
import pickle
import urllib
import io
import gzip
import urllib.request


class DBSCANAnalyzer:
    def __init__(self, dataset_name: str, cache_dir: str = "cache"):
        """
        Initialize the DBSCAN analyzer with dataset name and cache directory.

        Args:
            dataset_name: Name of the dataset (worms_2 or worms_64)
            cache_dir: Directory to store cached results
        """
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Create subdirectories for different result types
        self.clusters_cache = self.cache_dir / "clusters"
        self.plots_cache = self.cache_dir / "plots"
        self.clusters_cache.mkdir(exist_ok=True)
        self.plots_cache.mkdir(exist_ok=True)

        # Load data
        self.X, self.ref_labels = self._load_data()

        # Scale data
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Download and load the dataset and reference labels.

        Returns:
            Tuple of (data, reference_labels)
        """
        base_url = "https://raw.githubusercontent.com/gagolews/clustering-data-v1/master/sipu"

        # Download data if not cached
        data_path = self.cache_dir / f"{self.dataset_name}.data.gz"
        labels_path = self.cache_dir / f"{self.dataset_name}.labels0.gz"

        if not data_path.exists():
            data_response = urllib.request.urlopen(f"{base_url}/{self.dataset_name}.data.gz")
            compressed_data = io.BytesIO(data_response.read())
            with gzip.GzipFile(fileobj=compressed_data) as f:
                data_str = f.read().decode("utf-8")
                data = pd.read_csv(io.StringIO(data_str), header=None, sep=r"\s+")
            data.to_csv(data_path, header=False, index=False)

        if not labels_path.exists():
            labels_response = urllib.request.urlopen(f"{base_url}/{self.dataset_name}.labels0.gz")
            compressed_labels = io.BytesIO(labels_response.read())
            with gzip.GzipFile(fileobj=compressed_labels) as f:
                labels_str = f.read().decode("utf-8")
                labels = pd.read_csv(io.StringIO(labels_str), header=None, sep=r"\s+")
            labels.to_csv(labels_path, header=False, index=False)

        # Load data and labels
        X = pd.read_csv(data_path, header=None).values
        ref_labels = pd.read_csv(labels_path, header=None).values.ravel()

        return X, ref_labels

    def find_initial_parameters(self) -> Tuple[float, int]:
        """
        Find initial DBSCAN parameters using k-distance graph and dimensionality.

        Returns:
            Tuple of (eps, min_samples)
        """
        # Calculate min_samples based on dimensionality
        min_samples = 2 * self.X.shape[1]  # Rule of thumb: 2 * dimensions

        # Find optimal eps using k-distance graph
        n_neighbors = min_samples
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(self.X_scaled)
        distances, _ = nbrs.kneighbors(self.X_scaled)

        # Sort distances to the k-th nearest neighbor
        k_distances = np.sort(distances[:, -1])

        # Find knee point
        x = range(len(k_distances))
        kneedle = KneeLocator(x, k_distances, S=1.0, curve="convex", direction="increasing")
        eps = k_distances[kneedle.knee]

        # Plot k-distance graph
        plt.figure(figsize=(10, 6))
        plt.plot(x, k_distances)
        plt.axvline(x=kneedle.knee, color="r", linestyle="--", label="Knee point")
        plt.axhline(y=eps, color="r", linestyle="--")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"Distance to {n_neighbors}-th nearest neighbor")
        plt.title(f"K-distance graph for {self.dataset_name}")
        plt.legend()
        plt.savefig(self.plots_cache / f"{self.dataset_name}_kdistance.png")
        plt.close()

        return eps, min_samples

    def generate_parameter_grid(self, initial_eps: float, initial_min_samples: int) -> List[Dict]:
        """
        Generate a grid of parameters to search based on initial estimates.

        Args:
            initial_eps: Initial epsilon value
            initial_min_samples: Initial minimum samples value

        Returns:
            List of parameter dictionaries
        """
        eps_range = np.linspace(initial_eps * 0.5, initial_eps * 1.5, 10)
        min_samples_range = range(max(2, int(initial_min_samples * 0.5)), int(initial_min_samples * 1.5))

        return [{"eps": eps, "min_samples": ms} for eps in eps_range for ms in min_samples_range]

    def _get_cache_key(self, params: Dict) -> str:
        """Generate a cache key for given parameters."""
        return f"{self.dataset_name}_eps{params['eps']:.3f}_ms{params['min_samples']}"

    def run_dbscan(self, params: Dict) -> Tuple[np.ndarray, float, float, float]:
        """
        Run DBSCAN with given parameters and calculate validity metrics.

        Args:
            params: Dictionary with 'eps' and 'min_samples'

        Returns:
            Tuple of (labels, silhouette_score, ari_score, ami_score)
        """
        cache_key = self._get_cache_key(params)
        cache_file = self.clusters_cache / f"{cache_key}.pkl"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # Run DBSCAN
        dbscan = DBSCAN(**params).fit(self.X_scaled)
        labels = dbscan.labels_

        # Calculate metrics
        sil_score = silhouette_score(self.X_scaled, labels) if len(np.unique(labels)) > 1 else -1
        ari_score = adjusted_rand_score(self.ref_labels, labels)
        ami_score = adjusted_mutual_info_score(self.ref_labels, labels)

        # Cache results
        results = (labels, sil_score, ari_score, ami_score)
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)

        return results

    def find_best_parameters(self, param_grid: List[Dict]) -> Tuple[Dict, np.ndarray]:
        """
        Find the best parameters from the grid based on silhouette score.

        Args:
            param_grid: List of parameter dictionaries

        Returns:
            Tuple of (best_params, best_labels)
        """
        best_score = -1
        best_params = None
        best_labels = None

        results = []
        for params in param_grid:
            labels, sil_score, ari_score, ami_score = self.run_dbscan(params)
            results.append(
                {
                    "eps": params["eps"],
                    "min_samples": params["min_samples"],
                    "silhouette": sil_score,
                    "ari": ari_score,
                    "ami": ami_score,
                }
            )

            if sil_score > best_score:
                best_score = sil_score
                best_params = params
                best_labels = labels

        # Save parameter search results
        pd.DataFrame(results).to_csv(self.cache_dir / f"{self.dataset_name}_parameter_search.csv", index=False)

        return best_params, best_labels

    def visualize_clusters(self, labels: np.ndarray, title: str):
        """
        Visualize clustering results.

        Args:
            labels: Cluster labels
            title: Plot title
        """
        plt.figure(figsize=(12, 5))

        # For 2D data, create scatter plot
        if self.X.shape[1] == 2:
            plt.subplot(121)
            scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap="tab20")
            plt.colorbar(scatter)
            plt.title(f"Clustering Results\n{title}")

            plt.subplot(122)
            scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.ref_labels, cmap="tab20")
            plt.colorbar(scatter)
            plt.title("Reference Labels")

        # For high-dimensional data, create dimension reduction plot (using first few dimensions)
        else:
            plt.subplot(121)
            plt.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap="tab20")
            plt.title(f"First 2 Dimensions\n{title}")

            plt.subplot(122)
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.ref_labels, cmap="tab20")
            plt.title("Reference Labels\nFirst 2 Dimensions")

        plt.tight_layout()
        plt.savefig(self.plots_cache / f"{self.dataset_name}_clusters.png")
        plt.close()


def main():
    # Process both datasets
    for dataset_name in ["worms_2", "worms_64"]:
        print(f"\nProcessing {dataset_name}...")

        # Initialize analyzer
        analyzer = DBSCANAnalyzer(dataset_name)

        # Find initial parameters
        print("Finding initial parameters...")
        initial_eps, initial_min_samples = analyzer.find_initial_parameters()
        print(f"Initial parameters: eps={initial_eps:.3f}, min_samples={initial_min_samples}")

        # Generate parameter grid
        param_grid = analyzer.generate_parameter_grid(initial_eps, initial_min_samples)
        print(f"Generated {len(param_grid)} parameter combinations to test")

        # Find best parameters
        print("Finding best parameters...")
        best_params, best_labels = analyzer.find_best_parameters(param_grid)
        print(f"Best parameters: {best_params}")

        # Visualize results
        print("Visualizing results...")
        analyzer.visualize_clusters(
            best_labels, f"eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}"
        )

        print(f"Results saved in {analyzer.cache_dir}")


if __name__ == "__main__":
    main()
