import gc
import gzip
import io
import json
import os
import time
import urllib.request
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import init
from kneed import KneeLocator
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from dataclasses import dataclass

# from cuml.cluster import HDBSCAN

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class DataManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()

    def get_data(self, filename: str) -> pd.DataFrame:
        """Load data from local storage or download if not available"""
        data_path = self.base_dir / f"{filename}.pkl"

        if data_path.exists():
            self.console.print(f"[green]Loading {filename} from local storage[/green]")
            return pd.read_pickle(data_path)

        self.console.print(f"[yellow]Downloading {filename} from repository[/yellow]")
        data = self._download_data(filename)
        data.to_pickle(data_path)
        return data

    def _download_data(self, filename: str) -> pd.DataFrame:
        """Download data from repository"""
        url = f"https://raw.githubusercontent.com/gagolews/clustering-data-v1/master/sipu/{filename}.data.gz"

        try:
            response = urllib.request.urlopen(url)
            compressed_data = io.BytesIO(response.read())

            with gzip.GzipFile(fileobj=compressed_data) as f:
                data_str = f.read().decode("utf-8")
                data = pd.read_csv(io.StringIO(data_str), header=None, sep=r"\s+")

            return data

        except Exception as e:
            self.console.print(f"[red]Error downloading data: {e}[/red]")
            raise


@dataclass
class EvaluationResults:
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    n_clusters: int
    n_noise: int

    def to_dict(self):
        return {
            "silhouette_score": self.silhouette,
            "calinski_harabasz_score": self.calinski_harabasz,
            "davies_bouldin_score": self.davies_bouldin,
            "n_clusters": self.n_clusters,
            "n_noise_points": self.n_noise,
        }


class ClusteringMethods:
    def __init__(self, config: dict):
        self.config = config

    def dbscan_clustering(self, data: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """Perform DBSCAN clustering"""
        dbscan_params = self.config["clustering"]["dbscan"]
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            leaf_size=dbscan_params["leaf_size"],
            algorithm=dbscan_params["algorithm"],
            n_jobs=dbscan_params["n_jobs"],
        )
        return clusterer.fit_predict(data)

    def hdbscan_clustering(self, data: np.ndarray) -> np.ndarray:
        """Perform HDBSCAN clustering"""
        hdbscan_params = self.config["clustering"]["hdbscan"]
        clusterer = HDBSCAN(
            min_cluster_size=hdbscan_params["min_cluster_size"],
            min_samples=hdbscan_params["min_samples"],
            metric=hdbscan_params["metric"],
            cluster_selection_method=hdbscan_params["cluster_selection_method"],
            alpha=hdbscan_params["alpha"],
        )
        return clusterer.fit_predict(data)


class ClusteringAnalysis:
    def __init__(self, config_path: str = "config.json", output_dir: str = "clustering_results"):
        """Initialize the clustering analysis with configuration"""
        # Load configuration
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Initialize components
        self.data_manager = DataManager(Path(self.config["data"]["base_dir"]))
        self.clustering_methods = ClusteringMethods(self.config)
        # Initialize colorama for Windows PowerShell
        init()

        # Initialize Rich console
        self.console = Console()

        # Set up output directory
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.timestamp
        self.setup_directories()

        # Initialize results storage
        self.results = {}

        # Configure plotting style
        try:
            plt.style.use("seaborn-v0_8")
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")
            except OSError:
                self.console.print("[yellow]Warning: Could not set seaborn style. Using default style.[/yellow]")

        # Set default plot settings
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "axes.grid": True,
                "grid.alpha": 0.3,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "axes.titleweight": "bold",
                "lines.linewidth": 2,
                "scatter.edgecolors": "none",
                "font.size": 10,
                "font.family": "sans-serif",
            }
        )

        # Save configuration
        self.config = {
            "random_seed": RANDOM_SEED,
        }
        self.save_config()

    def setup_directories(self):
        """Create necessary directories for output"""
        directories = [self.run_dir, self.run_dir / "plots", self.run_dir / "data", self.run_dir / "metrics"]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        self.console.print(f"[green]Created output directories in {self.run_dir}[/green]")

    def save_config(self):
        """Save configuration to JSON file"""
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def save_results(self, dataset_name):
        """Save results for a specific dataset"""
        if dataset_name in self.results:
            results_path = self.run_dir / "metrics" / f"{dataset_name}_results.json"
            with open(results_path, "w") as f:
                # Convert numpy types to native Python types for JSON serialization
                results_dict = {
                    k: float(v) if isinstance(v, np.floating) else v for k, v in self.results[dataset_name].items()
                }
                json.dump(results_dict, f, indent=4)

    def save_plot(self, fig, name):
        """Save a matplotlib figure"""
        plot_path = self.run_dir / "plots" / f"{name}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.console.print(f"[green]Saved plot: {plot_path}[/green]")

    def visualize_clustering_comparison(self, data, labels_dbscan, labels_hdbscan, dataset_name):
        plt.clf()
        plt.close("all")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tuple(self.config["visualization"]["figure_size"]))

        scatter1 = ax1.scatter(
            data[:, 0],
            data[:, 1],
            c=labels_dbscan,
            cmap=self.config["visualization"]["cmap"],
            s=self.config["visualization"]["point_size"],
        )
        ax1.set_title("DBSCAN Clustering")
        plt.colorbar(scatter1, ax=ax1)

        # HDBSCAN plot
        scatter2 = ax2.scatter(
            data[:, 0],
            data[:, 1],
            c=labels_hdbscan,
            cmap=self.config["visualization"]["cmap"],
            s=self.config["visualization"]["point_size"],
        )
        ax2.set_title("HDBSCAN Clustering")
        plt.colorbar(scatter2, ax=ax2)

        plt.tight_layout()
        self.save_plot(fig, f"{dataset_name}_clustering_comparison")

    def visualize_dimension_reduction(self, data: np.ndarray, labels: np.ndarray, dataset_name: str, method: str):
        """Create dimensionality reduction visualization using both PCA and t-SNE"""
        if data.shape[1] <= 2:
            return

        plt.clf()
        plt.close("all")

        try:
            # Perform dimensionality reduction
            with self.progress_ctx() as progress:
                task1 = progress.add_task("[cyan]Computing PCA...", total=1)
                pca = PCA(n_components=2, random_state=RANDOM_SEED)
                data_pca = pca.fit_transform(data)
                progress.update(task1, advance=1)

                task2 = progress.add_task("[cyan]Computing t-SNE...", total=1)
                tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
                data_tsne = tsne.fit_transform(data)
                progress.update(task2, advance=1)

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tuple(self.config["visualization"]["figure_size"]))

            scatter1 = ax1.scatter(
                data_pca[:, 0],
                data_pca[:, 1],
                c=labels,
                cmap=self.config["visualization"]["cmap"],
                s=self.config["visualization"]["point_size"],
            )
            ax1.set_title(f"{method} - PCA Projection")
            plt.colorbar(scatter1, ax=ax1)

            scatter2 = ax2.scatter(
                data_tsne[:, 0],
                data_tsne[:, 1],
                c=labels,
                cmap=self.config["visualization"]["cmap"],
                s=self.config["visualization"]["point_size"],
            )
            ax2.set_title(f"{method} - t-SNE Projection")
            plt.colorbar(scatter2, ax=ax2)

            plt.tight_layout()
            self.save_plot(fig, f"{dataset_name}_{method.lower()}_dim_reduction")

        except Exception as e:
            self.console.print(f"[red]Error in dimension reduction visualization: {str(e)}[/red]")

    def load_data(self, filename):
        """Load gzipped data from GitHub repository with progress bar"""
        url = f"https://raw.githubusercontent.com/gagolews/clustering-data-v1/master/sipu/{filename}.data.gz"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            try:
                task = progress.add_task(f"[cyan]Downloading {filename}...", total=None)

                response = urllib.request.urlopen(url)
                compressed_data = io.BytesIO(response.read())
                progress.update(task, completed=50)

                with gzip.GzipFile(fileobj=compressed_data) as f:
                    data_str = f.read().decode("utf-8")
                    data = pd.read_csv(io.StringIO(data_str), header=None, sep=r"\s+")

                progress.update(task, completed=100)

                # Save raw data
                data_path = self.run_dir / "data" / f"{filename}_raw.pkl"
                data.to_pickle(data_path)
                self.console.print(f"[green]✓ Successfully loaded and saved {filename} dataset[/green]")

                return data

            except Exception as e:
                self.console.print(f"[red]Error loading data: {e}[/red]")
                return None

    def estimate_dbscan_parameters(self, data, dataset_name):
        """
        Estimate optimal DBSCAN parameters using the k-distance graph method
        """
        dimensions = data.shape[1]

        # Step 1: Estimate MinPts based on dimensionality
        if dimensions == 2:
            min_samples = 4  # Default for 2D (Ester et al., 1996)
            self.console.print("[cyan]2D dataset detected - using default MinPts = 4[/cyan]")
        else:
            min_samples = 2 * dimensions  # For higher dimensions (Sander et al., 1998)
            self.console.print(f"[cyan]{dimensions}D dataset detected - using MinPts = 2*dim = {min_samples}[/cyan]")

        # Step 2: Calculate k-nearest neighbors distances
        self.console.print(f"[cyan]Calculating {min_samples}-nearest neighbors distances...[/cyan]")
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(data)
        distances, _ = neighbors_fit.kneighbors(data)

        # Sort distances in ascending order
        k_distances = np.sort(distances[:, min_samples - 1])  # Exclude self-distance

        x_points = np.arange(len(k_distances))

        # Step 3: Find elbow point using maximum curvature
        kneedle = KneeLocator(
            x_points, k_distances, S=1.0, curve="convex", direction="increasing"  # Sensitivity parameter
        )

        eps = k_distances[kneedle.knee]
        elbow_idx = kneedle.knee

        # Step 4: Plot k-distance graph
        try:
            # Create new figure
            plt.clf()
            plt.close("all")
            kneedle.plot_knee_normalized()
            plot_path = self.run_dir / "plots" / f"{dataset_name}_knee_norm.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close("all")
            kneedle.plot_knee()
            plot_path = self.run_dir / "plots" / f"{dataset_name}_knee.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close("all")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Full plot
            ax1.plot(distances, "b-")
            ax1.axhline(y=eps, color="r", linestyle="--", label=f"Estimated eps = {eps:.2f}")
            ax1.set_title("K-distance Graph")
            ax1.set_xlabel("Points")
            ax1.set_ylabel("k-distance")
            ax1.legend()

            # Zoomed plot around the elbow
            zoom_range = slice(max(0, elbow_idx - 100), min(len(distances), elbow_idx + 100))
            ax2.plot(distances, "b-")
            ax2.set_xlim(zoom_range.start, zoom_range.stop)
            ax2.axhline(y=eps, color="r", linestyle="--", label=f"Estimated eps = {eps:.2f}")
            ax2.set_title("K-distance Graph (Zoomed)")
            ax2.set_xlabel("Points")
            ax2.set_ylabel("k-distance")
            ax2.legend()

            plt.tight_layout()

            # Save plot
            plot_path = self.run_dir / "plots" / f"{dataset_name}_kdistance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close("all")

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not create k-distance plot: {str(e)}[/yellow]")

        # Save estimation results
        estimation_results = {
            "dimensions": dimensions,
            "min_samples": min_samples,
            "eps": float(eps),
            "elbow_idx": int(elbow_idx),
            "distances": distances.tolist(),  # Save distances for possible later plotting
        }

        results_path = self.run_dir / "metrics" / f"{dataset_name}_parameter_estimation.json"
        with open(results_path, "w") as f:
            json.dump(estimation_results, f, indent=4)

        # Print results table
        table = Table(title="Estimated DBSCAN Parameters", show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Method", style="yellow")

        table.add_row("MinPts", str(min_samples), f"{'Default for 2D' if dimensions == 2 else '2 * dimensions'}")
        table.add_row("Epsilon (ε)", f"{eps:.3f}", "K-distance graph elbow point")

        self.console.print(table)

        return min_samples, eps, estimation_results

    def perform_clustering(self, data, eps=0.3, min_samples=5):
        """Perform DBSCAN clustering on the data"""
        # Remove the status context manager since this is called within a progress bar
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
        )
        clusters = dbscan.fit_predict(data_scaled)

        return clusters, data_scaled

    def evaluate_clustering(self, data, labels):
        """Calculate clustering quality metrics"""
        if len(np.unique(labels)) < 2:
            return None, None

        # Remove the status context manager
        silhouette = silhouette_score(data, labels)
        calinski = calinski_harabasz_score(data, labels)

        return silhouette, calinski

    def optimize_parameters(self, data, dataset_name):
        """
        Optimize DBSCAN parameters using estimated values as starting point
        """
        # First, get estimated parameters
        min_samples, eps, estimation_results = self.estimate_dbscan_parameters(data, dataset_name)

        # Define search ranges around estimated values
        eps_range = [0.9 * eps, eps, 1.1 * eps]  # [0.9 * eps, eps, 1.1* eps]  # Search around estimated eps
        min_samples_range = [
            min_samples - 1,
            min_samples,
            min_samples + 1,
        ]  # [min_samples - 1,min_samples,min_samples + 1,]
        # Store parameter ranges in config
        self.config[f"{dataset_name}_parameter_ranges"] = {
            "eps_range": list(eps_range),
            "min_samples_range": list(min_samples_range),
        }
        self.save_config()

        # Perform grid search around estimated values
        best_score = -1
        best_params = None
        results = []
        total_iterations = len(eps_range) * len(min_samples_range)

        self.console.print("\n[cyan]Fine-tuning parameters around estimated values...[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"[cyan]Optimizing parameters for {dataset_name}...", total=total_iterations)

            for eps_val in eps_range:
                for min_samples_val in min_samples_range:
                    progress.update(task, description=f"[cyan]Testing eps={eps_val:.2f}, min_samples={min_samples_val}")

                    clusters, _ = self.perform_clustering(data, eps_val, min_samples_val)

                    if len(np.unique(clusters)) >= 2:
                        silhouette, calinski = self.evaluate_clustering(data, clusters)

                        if silhouette and silhouette > best_score:
                            best_score = silhouette
                            best_params = (eps_val, min_samples_val)
                            progress.update(
                                task,
                                description=f"[green]New best: eps={eps_val:.2f}, min_samples={min_samples_val}, score={silhouette:.3f}",
                            )

                        results.append(
                            {
                                "eps": float(eps_val),
                                "min_samples": int(min_samples_val),
                                "silhouette": float(silhouette) if silhouette else None,
                                "calinski": float(calinski) if calinski else None,
                                "n_clusters": int(len(np.unique(clusters[clusters >= 0]))),
                            }
                        )

                    progress.update(task, advance=1)

        # Save optimization results
        optimization_results = {
            "estimated_parameters": estimation_results,
            "grid_search_results": results,
            "best_parameters": {
                "eps": float(best_params[0]),
                "min_samples": int(best_params[1]),
                "silhouette_score": float(best_score),
            },
        }

        results_path = self.run_dir / "metrics" / f"{dataset_name}_parameter_optimization.json"
        with open(results_path, "w") as f:
            json.dump(optimization_results, f, indent=4)

        return best_params

    def evaluate_and_store_results(
        self, data: np.ndarray, dbscan_labels: np.ndarray, hdbscan_labels: np.ndarray, filename: str
    ) -> None:
        """Evaluate clustering results and store metrics"""

        def evaluate_clustering(labels: np.ndarray) -> EvaluationResults:
            # Skip evaluation if all points are noise
            if len(np.unique(labels[labels >= 0])) <= 1:
                return EvaluationResults(
                    silhouette=0.0,
                    calinski_harabasz=0.0,
                    davies_bouldin=0.0,
                    n_clusters=0,
                    n_noise=np.sum(labels == -1),
                )

            return EvaluationResults(
                silhouette=silhouette_score(data, labels, metric="euclidean"),
                calinski_harabasz=calinski_harabasz_score(data, labels),
                davies_bouldin=davies_bouldin_score(data, labels),
                n_clusters=len(np.unique(labels[labels >= 0])),
                n_noise=np.sum(labels == -1),
            )

        # Evaluate both methods
        dbscan_results = evaluate_clustering(dbscan_labels)
        hdbscan_results = evaluate_clustering(hdbscan_labels)

        # Store results
        self.results[filename] = {
            "dbscan": {
                "optimal_eps": float(self.best_params[filename][0]),
                "optimal_min_samples": int(self.best_params[filename][1]),
                **dbscan_results.to_dict(),
            },
            "hdbscan": hdbscan_results.to_dict(),
        }

        # Create comparison table
        table = Table(title=f"Clustering Results - {filename}", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("DBSCAN", style="green")
        table.add_column("HDBSCAN", style="blue")

        metrics = [
            ("Silhouette Score", "silhouette_score"),
            ("Calinski-Harabasz Score", "calinski_harabasz_score"),
            ("Davies-Bouldin Score", "davies_bouldin_score"),
            ("Number of Clusters", "n_clusters"),
            ("Noise Points", "n_noise_points"),
        ]

        for metric_name, metric_key in metrics:
            dbscan_value = self.results[filename]["dbscan"][metric_key]
            hdbscan_value = self.results[filename]["hdbscan"][metric_key]
            table.add_row(
                metric_name,
                f"{dbscan_value:.4f}" if isinstance(dbscan_value, float) else str(dbscan_value),
                f"{hdbscan_value:.4f}" if isinstance(hdbscan_value, float) else str(hdbscan_value),
            )

        # Add DBSCAN specific parameters
        table.add_row("Optimal Epsilon", f"{self.results[filename]['dbscan']['optimal_eps']:.4f}", "N/A")
        table.add_row("Optimal Min Samples", str(self.results[filename]["dbscan"]["optimal_min_samples"]), "N/A")

        self.console.print(table)

    def analyze_dataset(self, filename, is_2d=True):
        """Main analysis function for a dataset"""
        self.console.rule(f"[bold blue]Analyzing {filename}")

        # Load data
        data = self.data_manager.get_data(filename)
        if data is None:
            return
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        self.console.print(f"\n[bold cyan]Dataset Shape:[/bold cyan] {data.shape}")

        # Parameter optimization
        best_params = self.optimize_parameters(data_scaled, filename)

        if best_params is None:
            self.console.print("[red]Could not find suitable parameters[/red]")
            return

        self.console.print("\n[bold green]Optimal Parameters Found:[/bold green]")
        self.console.print(f"eps = {best_params[0]:.2f}")
        self.console.print(f"min_samples = {best_params[1]}")

        # Perform final clustering with optimal parameters
        self.console.print("\n[cyan]Performing final clustering with optimal parameters...[/cyan]")
        dbscan_labels = self.clustering_methods.dbscan_clustering(data_scaled, best_params[0], best_params[1])
        hdbscan_labels = self.clustering_methods.hdbscan_clustering(data_scaled)

        # Evaluate clustering
        self.console.print("[cyan]Evaluating final clustering...[/cyan]")
        self.evaluate_and_store_results(data_scaled, dbscan_labels, hdbscan_labels, filename)

        # # Store results
        # self.results[filename] = {
        #     "optimal_eps": float(best_params[0]),
        #     "optimal_min_samples": int(best_params[1]),
        #     "silhouette_score": float(silhouette),
        #     "calinski_harabasz_score": float(calinski),
        #     "n_clusters": int(len(np.unique(clusters[clusters >= 0]))),
        #     "n_noise_points": int(np.sum(clusters == -1)),
        # }

        # # Save results
        # self.save_results(filename)

        # # Create results table
        # table = Table(title="Clustering Results", show_header=True, header_style="bold magenta")
        # table.add_column("Metric", style="cyan")
        # table.add_column("Value", style="green")
        # for key, value in self.results[filename].items():
        #     table.add_row(key, f"{value}")
        # self.console.print(table)

        # # Visualize results
        # self.console.print("\n[cyan]Generating visualizations...[/cyan]")
        # if is_2d:
        #     self.visualize_2d_clustering(data_scaled, clusters, f"DBSCAN Clustering Results - {filename}", filename)
        # else:
        #     self.visualize_high_dim_clustering(
        #         data_scaled, clusters, f"DBSCAN Clustering Results - {filename}", filename
        #     )

        if is_2d:
            self.visualize_clustering_comparison(data_scaled, dbscan_labels, hdbscan_labels, filename)
        else:
            self.visualize_dimension_reduction(data_scaled, dbscan_labels, filename, "DBSCAN")
            self.visualize_dimension_reduction(data_scaled, hdbscan_labels, filename, "HDBSCAN")

    def visualize_2d_clustering(self, data, labels, title, dataset_name):
        """Visualize clustering results for 2D data"""
        plt.clf()
        plt.close("all")

        fig, ax = plt.subplots(figsize=tuple(self.config["visualization"]["figure_size"]))
        scatter = ax.scatter(
            data[:, 0],
            data[:, 1],
            c=labels,
            cmap=self.config["visualization"]["cmap"],
            s=self.config["visualization"]["point_size"],
        )
        plt.colorbar(scatter)
        ax.set_title(title, pad=20, fontsize=14, fontweight="bold")
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

    def visualize_high_dim_clustering(self, data, labels, title, dataset_name):
        """Visualize clustering results for high-dimensional data using PCA"""
        try:
            plt.clf()
            plt.close("all")

            # Apply PCA
            pca = PCA(n_components=2, random_state=RANDOM_SEED)
            data_2d = pca.fit_transform(data)

            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap="viridis", alpha=0.6)
            plt.colorbar(scatter)
            ax.set_title(f"{title}\n(PCA visualization)", pad=20, fontsize=14, fontweight="bold")
            ax.set_xlabel("First Principal Component", fontsize=12)
            ax.set_ylabel("Second Principal Component", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Save plot
            plot_path = self.run_dir / "plots" / f"{dataset_name}_pca_clustering.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close("all")

            # Save PCA results
            pca_results = {
                "explained_variance_ratio": list(pca.explained_variance_ratio_),
                "cumulative_variance_ratio": list(np.cumsum(pca.explained_variance_ratio_)),
            }
            with open(self.run_dir / "metrics" / f"{dataset_name}_pca_results.json", "w") as f:
                json.dump(pca_results, f, indent=4)

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not create PCA visualization: {str(e)}[/yellow]")

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        # Create fancy header
        self.create_fancy_header()

        # Analyze both datasets
        self.analyze_dataset("worms_2", is_2d=True)
        self.analyze_dataset("worms_64", is_2d=False)

        # Save final summary
        self.save_final_summary()

    def create_fancy_header(self):
        """Create a fancy header for the analysis"""
        header = Text()
        header.append("🔍 ", style="bold yellow")
        header.append("Data Clustering Analysis", style="bold blue")
        header.append(" 🔍", style="bold yellow")
        self.console.print(Panel(header, border_style="blue"))
        self.console.print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def save_final_summary(self):
        """Save final summary of all results"""
        summary_path = self.run_dir / "final_summary.json"
        summary = {"timestamp": self.timestamp, "results": self.results, "config": self.config}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        self.console.print(f"\n[green]Final summary saved to: {summary_path}[/green]")
        self.console.print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main function to run the analysis"""
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Create and run analysis
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
