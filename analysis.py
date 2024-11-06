import gzip
import io
import json
import os
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import init
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class ClusteringAnalysis:
    def __init__(self, output_dir="clustering_results"):
        """Initialize the clustering analysis with output directory setup"""
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
                    data = pd.read_csv(io.StringIO(data_str), header=None, sep="\s+")

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
        distances = np.sort(distances[:, 1])  # Exclude self-distance

        # Step 3: Find elbow point using maximum curvature
        first_derivative = np.gradient(distances)
        second_derivative = np.gradient(first_derivative)
        curvature = np.abs(second_derivative) / (1 + first_derivative**2) ** 1.5
        elbow_idx = np.argmax(curvature)
        eps = distances[elbow_idx]

        # Step 4: Plot k-distance graph
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
        plt.close()

        # Save estimation results
        estimation_results = {
            "dimensions": dimensions,
            "min_samples": min_samples,
            "eps": float(eps),
            "elbow_idx": int(elbow_idx),
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
        eps_range = np.linspace(eps * 0.5, eps * 1.5, 20)  # [0.9 * eps, eps, 1.1, eps]  # Search around estimated eps
        min_samples_range = range(
            max(2, int(min_samples * 0.5)), int(min_samples * 1.5)
        )  # [min_samples - 1,min_samples,min_samples + 1,]

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

    def analyze_dataset(self, filename, is_2d=True):
        """Main analysis function for a dataset"""
        self.console.rule(f"[bold blue]Analyzing {filename}")

        # Load data
        data = self.load_data(filename)
        if data is None:
            return

        self.console.print(f"\n[bold cyan]Dataset Shape:[/bold cyan] {data.shape}")

        # Parameter optimization
        best_params = self.optimize_parameters(data, filename)

        if best_params is None:
            self.console.print("[red]Could not find suitable parameters[/red]")
            return

        self.console.print("\n[bold green]Optimal Parameters Found:[/bold green]")
        self.console.print(f"eps = {best_params[0]:.2f}")
        self.console.print(f"min_samples = {best_params[1]}")

        # Perform final clustering with optimal parameters
        self.console.print("\n[cyan]Performing final clustering with optimal parameters...[/cyan]")
        clusters, data_scaled = self.perform_clustering(data, best_params[0], best_params[1])

        # Evaluate clustering
        self.console.print("[cyan]Evaluating final clustering...[/cyan]")
        silhouette, calinski = self.evaluate_clustering(data_scaled, clusters)

        # Store results
        self.results[filename] = {
            "optimal_eps": float(best_params[0]),
            "optimal_min_samples": int(best_params[1]),
            "silhouette_score": float(silhouette),
            "calinski_harabasz_score": float(calinski),
            "n_clusters": int(len(np.unique(clusters[clusters >= 0]))),
            "n_noise_points": int(np.sum(clusters == -1)),
        }

        # Save results
        self.save_results(filename)

        # Create results table
        table = Table(title="Clustering Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for key, value in self.results[filename].items():
            table.add_row(key, f"{value}")
        self.console.print(table)

        # Visualize results
        self.console.print("\n[cyan]Generating visualizations...[/cyan]")
        if is_2d:
            self.visualize_2d_clustering(data_scaled, clusters, f"DBSCAN Clustering Results - {filename}", filename)
        else:
            self.visualize_high_dim_clustering(
                data_scaled, clusters, f"DBSCAN Clustering Results - {filename}", filename
            )

    def visualize_2d_clustering(self, data, labels, title, dataset_name):
        """Visualize clustering results for 2D data"""
        with self.console.status("[cyan]Creating 2D visualization...", spinner="dots"):
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", alpha=0.6)
            plt.colorbar(scatter)
            ax.set_title(title, pad=20, fontsize=14, fontweight="bold")
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Y", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Save plot
            self.save_plot(fig, f"{dataset_name}_2d_clustering")
            plt.close(fig)

    def visualize_high_dim_clustering(self, data, labels, title, dataset_name):
        """Visualize clustering results for high-dimensional data using PCA"""
        with self.console.status("[cyan]Performing dimensionality reduction and visualization...", spinner="dots"):
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

            # Save plot and PCA results
            self.save_plot(fig, f"{dataset_name}_pca_clustering")
            plt.close(fig)

            # Save PCA explained variance ratios
            pca_results = {
                "explained_variance_ratio": list(pca.explained_variance_ratio_),
                "cumulative_variance_ratio": list(np.cumsum(pca.explained_variance_ratio_)),
            }
            with open(self.run_dir / "metrics" / f"{dataset_name}_pca_results.json", "w") as f:
                json.dump(pca_results, f, indent=4)

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
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
    analyzer = ClusteringAnalysis()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
