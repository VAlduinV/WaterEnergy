import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from src.CONSTANT.constant import *


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the Dunn index for the given cluster labels.

    Args:
        X (np.ndarray): Data points.
        labels (np.ndarray): Cluster labels.

    Returns:
        float: Dunn index value.
    """
    distances = pairwise_distances(X)  # Calculate pairwise distances
    unique_clusters = np.unique(labels)  # Get unique clusters

    min_intercluster = np.inf  # Initialize minimum intercluster distance
    max_intracluster = 0  # Initialize maximum intracluster distance

    for i in unique_clusters:
        for j in unique_clusters:
            if i != j:
                intercluster_distances = distances[labels == i, :][
                    :, labels == j
                ]  # Calculate intercluster distances
                if intercluster_distances.size > 0:
                    min_intercluster = min(
                        min_intercluster, np.min(intercluster_distances)
                    )  # Update minimum intercluster distance

        intracluster_distances = distances[labels == i, :][
            :, labels == i
        ]  # Calculate intracluster distances
        if intracluster_distances.size > 0:
            max_intracluster = max(
                max_intracluster, np.max(intracluster_distances)
            )  # Update maximum intracluster distance

    if max_intracluster == 0:
        return 0  # To avoid division by zero

    return min_intercluster / max_intracluster  # Return Dunn index


def evaluate_clusters_and_plot(X: pd.DataFrame, max_clusters: int):
    """
    Evaluates clustering for a range of cluster numbers and plots Dunn Index.

    Args:
        X (pd.DataFrame): Data points.
        max_clusters (int): Maximum number of clusters to evaluate.

    Returns:
        list: Dunn index values for each number of clusters.
    """
    dunn_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)  # Initialize KMeans
        labels = kmeans.fit_predict(X)  # Predict cluster labels
        dunn_score = dunn_index(X, labels)  # Calculate Dunn index
        dunn_scores.append(dunn_score)  # Append Dunn index to list
        print(f"Dunn Index for k={k}: {dunn_score}")

    plt.figure(figsize=FIG_SIZE)  # Create figure
    plt.plot(
        range(2, max_clusters + 1),
        dunn_scores,
        "o-",
        color="red",
        linewidth=4,
        markersize=12,
        markeredgewidth=4,
        markerfacecolor="pink",
        markeredgecolor="darkgreen",
        label="Dunn Index per Cluster",
    )  # Plot Dunn index
    plt.xlabel("Number of Clusters")  # Set x-label
    plt.ylabel("Dunn Index")  # Set y-label
    plt.title("Dunn Index for Various Numbers of Clusters")  # Set title
    plt.legend(
        title="Optimization of Smart-Villages Clusters via Dunn Index",
        title_fontsize=14,
        prop={"family": "Times New Roman", "size": 14, "weight": "bold"},
        loc="upper right",
        shadow=True,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor="darkgrey",
        edgecolor="yellow",
    )  # Add legend
    mplcyberpunk.add_glow_effects()  # Add glow effects
    plt.grid(True)  # Show grid
    plt.show()  # Show plot

    return dunn_scores
