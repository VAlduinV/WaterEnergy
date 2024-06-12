import umap
import seaborn as sns
from src.CONSTANT.constant import *


def display_cluster_umap(
        data: np.ndarray,
        predicted_clusters: np.ndarray,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
):
    """
    Apply UMAP dimensionality reduction on high-dimensional data and plot the results.

    Args:
        data (np.ndarray): High-dimensional data to be reduced.
        predicted_clusters (np.ndarray): Cluster labels for each point in data.
        n_neighbors (int): The size of local neighborhood used for manifold approximation.
        min_dist (float): The minimum distance apart that points are allowed to be in the low-dimensional representation.
        n_components (int): The number of dimensions to reduce to.

    Displays:
        Scatter plot of the data reduced to two dimensions.
    """
    predicted_clusters = predicted_clusters + 1  # Adjust cluster indices to start from 1
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
    )  # Initialize UMAP
    embedding = reducer.fit_transform(data)  # Transform data using UMAP

    plt.figure(figsize=FIG_SIZE)  # Create figure
    scatter = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=predicted_clusters,
        palette=sns.color_palette("hsv", len(set(predicted_clusters))),
        s=50,
        alpha=0.6,
        edgecolor="w",
    )  # Create scatter plot
    scatter.set_title("UMAP Projection of the Clustered Data",
                      fontdict={"family": "Times New Roman", 'fontsize': 14})  # Set title
    legend = plt.legend(
        loc="upper right",
        title="Cluster ID",
        prop={"family": "Times New Roman", "size": 14, "weight": "bold"},
        labelcolor="white",
        shadow=True,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor="darkgrey",
        edgecolor="red",
    )  # Add legend
    plt.xlabel("$Component_{1}$", fontdict={"family": "Times New Roman", 'fontsize': 14})  # Set x-label
    plt.ylabel("$Component_{2}$", fontdict={"family": "Times New Roman", 'fontsize': 14})  # Set y-label
    # Set legend title colors
    plt.setp(legend.get_title(), color="white")
    plt.show()  # Show plot
